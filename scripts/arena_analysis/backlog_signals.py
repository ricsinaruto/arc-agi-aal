"""In-run signals for the fixed-2x backlog, checked on a recorded run (hero-v2 2 h, 29 September; CPU only).

User decision (29 September, run 2 of docs/validation-20260927/hero-v2-next-runs.md): replace the roster-order level-0
backlog priority with an in-run signal (no per-game priors from earlier runs), stop no-op parks, and check whether a
small park grace near a progress signal is justified. This script evaluates candidate signals on the recorded
trajectories (docs/validation-20260927/hero-v2-analysis/scheduling.md for the findings it starts from):

- ``level0``: every level-0 game parked at the opening switch, with its opening evidence at the switch (gameplay
  tokens, actions, distinct boards on level 0 = the scheduler's ``frames_on_level``, effective actions = actions whose
  board differs from the board before, actions since the last new board) and what it did in phase 2 (cleared L1 or
  not; tokens from its first phase-2 admission to the clear). For each candidate order it reports the AUC for "clears
  L1 in phase 2", the mean backlog position of the converters (0 = first), and the positions of the quick converters
  (L1 within 10K tokens of re-admission), each overall and within a game family (clones share a family on c110).
- ``parks``: every level-budget park at level >= 1 with the same evidence on the level at the park (actions since the
  last new board, new boards in the last 10 / 20 actions, tokens since the last new board), and the tokens after
  re-admission to the next clear.
- ``noop``: parks re-admitted in the same scheduler tick (the "no-op" budget parks).

    PYTHONPATH=src uv run python scripts/arena_analysis/backlog_signals.py <trial dir> --config <comparison.json>
        [--out result.json]

Board keys follow the Duck adapter (``arena.duck``: sha1 of the observed board, keyed by the level): the last frame of
each logged observation. Holdout games are reported but used for no constant.
"""

import argparse
import bisect
import hashlib
import json
import statistics
from collections import defaultdict
from pathlib import Path

HOLDOUT = ("sb26", "re86", "s5i5", "su15", "dc22", "cd82", "g50t", "ls20", "tr87")
QUICK_TOKENS = 10_000


def board_key(frames):
    last = frames[-1] if frames and isinstance(frames[0], list) else frames
    return hashlib.sha1(json.dumps(last).encode()).hexdigest()


class Game:
    def __init__(self, raw, family, log_path, release_epoch):
        self.raw, self.id, self.family = raw, raw["game_id"], family
        self.index = raw["index"]
        self.clear_walls = [c["wall_seconds"] for c in raw["time_to_level"]]
        self.charges = [(c["wall_seconds"], c["completion_tokens"]) for c in raw["charges"] if c["purpose"] == "gameplay"]
        self.charge_walls = [w for w, _ in self.charges]
        cum, total = [], 0
        for _, tokens in self.charges:
            total += tokens
            cum.append(total)
        self.charge_cum = cum
        self.admissions = raw["admissions"]
        self.actions = []  # (wall, level before, key after, changed)
        previous_key, previous_level = None, 0
        if log_path.is_file():
            for line in log_path.open():
                row = json.loads(line)
                if row.get("kind") != "observation" or not row.get("frames"):
                    continue
                wall = row["time"] - release_epoch
                key = (row.get("levels", 0), board_key(row["frames"]))
                if row.get("action") is not None:
                    self.actions.append((wall, previous_level, key, key != previous_key))
                previous_key, previous_level = key, row.get("levels", 0)
        self.action_walls = [a[0] for a in self.actions]

    def level_at(self, wall):
        return bisect.bisect_left(self.clear_walls, wall)

    def tokens_at(self, wall):
        i = bisect.bisect_right(self.charge_walls, wall)
        return self.charge_cum[i - 1] if i else 0

    def evidence(self, wall):
        """The scheduler's view of the current level at ``wall``."""
        level = self.level_at(wall)
        start = self.clear_walls[level - 1] if level else 0.0
        seen, novel_walls, actions, effective, since_novel = set(), [], 0, 0, 0
        recent = []
        for a_wall, _, key, changed in self.actions[:bisect.bisect_right(self.action_walls, wall)]:
            if a_wall < start:
                continue
            actions += 1
            effective += bool(changed)
            new = key not in seen
            seen.add(key)
            recent.append(new)
            if new:
                novel_walls.append(a_wall)
                since_novel = 0
            else:
                since_novel += 1
        tokens = self.tokens_at(wall)
        level_tokens = tokens - (self.tokens_at(start) if level else 0)
        last_novel = novel_walls[-1] if novel_walls else None
        return {"level": level, "tokens": tokens, "level_tokens": level_tokens, "actions": actions,
                "boards": len(seen), "effective": effective,
                "effective_fraction": round(effective / actions, 3) if actions else None,
                "since_novel": since_novel, "new_last10": sum(recent[-10:]), "new_last20": sum(recent[-20:]),
                "tokens_since_novel": None if last_novel is None else tokens - self.tokens_at(last_novel)}

    def spans(self, wall_end):
        out, since, reason = [], None, None
        for e in self.admissions:
            if e["event"] == "admit":
                since, reason = e["wall_seconds"], e["reason"]
            elif e["event"] in ("park", "finish") and since is not None:
                out.append((since, e["wall_seconds"], reason, e["reason"]))
                since = None
        if since is not None:
            out.append((since, wall_end, reason, "wall_end"))
        return out


def load(trial_dir, config):
    trial_dir = Path(trial_dir)
    report = json.loads((trial_dir / "arena.json").read_text())
    names = json.loads(Path(config).read_text())["games"]
    release = None
    with (trial_dir / "admission.jsonl").open() as handle:
        for line in handle:
            row = json.loads(line)
            if row.get("wall_seconds") == 0.0 and row.get("epoch"):
                release = row["epoch"]
                break
    games = [Game(g, names[g["index"] % len(names)][:4], trial_dir / "games" / g["game_id"] / "log.jsonl", release)
             for g in report["games"]]
    return report, games


def auc(scores, labels):
    pos = [s for s, y in zip(scores, labels) if y]
    neg = [s for s, y in zip(scores, labels) if not y]
    if not pos or not neg:
        return None
    wins = sum((p > n) + 0.5 * (p == n) for p in pos for n in neg)
    return round(wins / (len(pos) * len(neg)), 3)


# Candidate level-0 orders: a key per game (lower = admitted first) from its switch-time evidence; roster is the
# current rule (readmissions tie -> roster index).
ORDERS = {
    "roster": lambda g, e: g.index,
    "most_boards": lambda g, e: (-e["boards"], g.index),
    "fewest_tokens": lambda g, e: (e["tokens"], g.index),
    "most_tokens": lambda g, e: (-e["tokens"], g.index),
    "most_actions": lambda g, e: (-e["actions"], g.index),
    "boards_per_10k_tokens": lambda g, e: (-e["boards"] / max(e["tokens"], 1), g.index),
    "effective_fraction": lambda g, e: (-(e["effective_fraction"] or 0), g.index),
    "fewest_since_novel": lambda g, e: (e["since_novel"], g.index),
    "new_last20": lambda g, e: (-e["new_last20"], g.index),
}


def bootstrap_auc(scores, labels, draws=2000, seed=0):
    """95% interval of the AUC over games resampled with replacement."""
    import random

    rng = random.Random(seed)
    pairs = list(zip(scores, labels))
    values = []
    for _ in range(draws):
        sample = [rng.choice(pairs) for _ in pairs]
        a = auc([s for s, _ in sample], [y for _, y in sample])
        if a is not None:
            values.append(a)
    values.sort()
    return [values[int(0.025 * len(values))], values[int(0.975 * len(values)) - 1]]


def level0(games, switch, wall_end):
    rows = []
    for g in games:
        parks = [a for a in g.admissions if a["event"] == "park" and a["reason"] == "opening_switch"]
        if not parks or parks[0]["levels"] != 0:
            continue
        e = g.evidence(switch)
        admits = [s for s in g.spans(wall_end) if s[0] > switch]
        first = admits[0][0] if admits else None
        l1 = g.clear_walls[0] if g.clear_walls else None
        cleared = l1 is not None and l1 > switch
        to_clear = None
        if cleared and first is not None and l1 >= first:
            to_clear = g.tokens_at(l1) - g.tokens_at(first)
        rows.append({"game": g.id, "family": g.family, "holdout": g.family in HOLDOUT, "index": g.index,
                     "evidence": e, "first_phase2_admission": first, "l1_wall": l1, "cleared_phase2": cleared,
                     "tokens_admission_to_l1": to_clear,
                     "quick": bool(to_clear is not None and to_clear <= QUICK_TOKENS)})
    result = {"games": len(rows), "cleared": sum(r["cleared_phase2"] for r in rows),
              "quick": sum(r["quick"] for r in rows), "orders": {}}
    for name, order in ORDERS.items():
        ranked = sorted(rows, key=lambda r: order(next(g for g in games if g.id == r["game"]), r["evidence"]))
        position = {r["game"]: i for i, r in enumerate(ranked)}
        score = [-position[r["game"]] for r in rows]
        within = []
        by_family = defaultdict(list)
        for r in rows:
            by_family[r["family"]].append(r)
        for members in by_family.values():
            a = auc([-position[r["game"]] for r in members], [r["cleared_phase2"] for r in members])
            if a is not None:
                within.append((a, len(members)))
        within_quick = []
        for members in by_family.values():
            a = auc([-position[r["game"]] for r in members], [r["quick"] for r in members])
            if a is not None:
                within_quick.append((a, len(members)))
        result["orders"][name] = {
            "auc_cleared": auc(score, [r["cleared_phase2"] for r in rows]),
            "auc_quick": auc(score, [r["quick"] for r in rows]),
            "auc_quick_95": bootstrap_auc(score, [r["quick"] for r in rows]),
            "auc_quick_within_family": (round(sum(a * n for a, n in within_quick) / sum(n for _, n in within_quick), 3)
                                        if within_quick else None),
            "auc_cleared_within_family": (round(sum(a * n for a, n in within) / sum(n for _, n in within), 3)
                                          if within else None),
            "families_with_both": len(within),
            "mean_position_cleared": round(statistics.mean(position[r["game"]] for r in rows if r["cleared_phase2"]), 1),
            "quick_positions": sorted(position[r["game"]] for r in rows if r["quick"]),
            "mean_position_quick": (round(statistics.mean(position[r["game"]] for r in rows if r["quick"]), 1)
                                    if any(r["quick"] for r in rows) else None),
            "first_20": [ranked[i]["game"] + ("*" if ranked[i]["cleared_phase2"] else "") for i in range(20)],
        }
    result["rows"] = rows
    return result


def parks(games, wall_end):
    rows = []
    for g in games:
        spans = g.spans(wall_end)
        for i, a in enumerate(g.admissions):
            if a["event"] != "park" or a["reason"] != "level_budget" or a["levels"] < 1:
                continue
            wall = a["wall_seconds"]
            nxt = next((s for s in spans if s[0] >= wall), None)
            readmit = nxt[0] if nxt else None
            clear = next((c for c in g.clear_walls if c > wall), None)
            to_clear = (g.tokens_at(clear) - g.tokens_at(readmit)) if (clear and readmit and clear >= readmit) else None
            rows.append({"game": g.id, "family": g.family, "holdout": g.family in HOLDOUT, "levels": a["levels"],
                         "wall": wall, "readmit": readmit,
                         "parked_seconds": None if readmit is None else round(readmit - wall, 1),
                         "noop": readmit is not None and readmit - wall < 1.5,
                         "evidence": g.evidence(wall), "tokens_readmission_to_clear": to_clear,
                         "next_clear_wall": clear})
    return rows


# --- trace-driven replay of the phase-2 backlog order ------------------------------------------------------------

NOOP_GAP = 0.0


def phase2_tenures(g, switch, wall_end):
    """The game's phase-2 tenures in order (no-op park + re-admission gaps merged): start, duration, tokens, the clears
    (offset in the tenure, level) and how the tenure ended."""
    spans = []
    for start, stop, reason_in, reason_out in g.spans(wall_end):
        if stop <= switch + 0.5 or reason_out == "opening_switch":
            continue  # the opening (parked at the switch, a millisecond after it)
        start = max(start, switch)
        if spans and start - spans[-1][1] < NOOP_GAP:
            spans[-1] = (spans[-1][0], stop, spans[-1][2], reason_out)
        else:
            spans.append((start, stop, reason_in, reason_out))
    out = []
    for start, stop, reason_in, reason_out in spans:
        clears = [(c - start, i + 1) for i, c in enumerate(g.clear_walls) if start <= c < stop + 1e-6]
        out.append({"start": start, "duration": stop - start, "tokens": g.tokens_at(stop) - g.tokens_at(start),
                    "clears": clears, "end": reason_out, "reason_in": reason_in})
    return out


def points(g_levels, level):
    return 100.0 * level / (g_levels * (g_levels + 1) / 2) / 110.0


def _auditions_first(w, signal):
    return (not w["pending"],) + (
        (1, -w["levels"], w["readmissions"]) if w["levels"] >= 1
        else (0, 0, -signal) if w["readmissions"] == 0
        else (2, w["readmissions"], -signal)) + (w["index"],)


def _progress_fruitless(w, signal):
    progress = w["pending"] or w["levels"] > w["admit_levels"]
    return (not progress, w["fruitless_on_level"], -w["levels"], -(signal if w["levels"] == 0 else 0), w["index"])


RULES = {
    # The chosen order: progress, fewest fruitless tenures on the level, more
    # levels, the level-0 tie-break, roster order.
    "progress-fruitless+tokens": lambda w: _progress_fruitless(w, w["level0_tokens"]),
    "progress-fruitless+roster": lambda w: _progress_fruitless(w, 0),
    # (key over the waiting game's state; lower first). pending = a clear that arrived while parked.
    "current": lambda w: (not w["pending"], w["readmissions"], -w["levels"], w["index"]),
    "tokens": lambda w: (not w["pending"], w["readmissions"], -w["levels"],
                         -(w["level0_tokens"] if w["levels"] == 0 else 0), w["index"]),
    "auditions-first+tokens": lambda w: (not w["pending"],) + (
        (1, -w["levels"], w["readmissions"]) if w["levels"] >= 1
        else (0, 0, -w["level0_tokens"]) if w["readmissions"] == 0
        else (2, w["readmissions"], -w["level0_tokens"])) + (w["index"],),
    "deep-first+tokens": lambda w: (not w["pending"],) + (
        (0, -w["levels"], w["readmissions"]) if w["levels"] >= 1
        else (1, 0, -w["level0_tokens"]) if w["readmissions"] == 0
        else (2, w["readmissions"], -w["level0_tokens"])) + (w["index"],),
    "auditions-first": lambda w: (not w["pending"],) + (
        (1, -w["levels"], w["readmissions"]) if w["levels"] >= 1
        else (0, 0, 0) if w["readmissions"] == 0 else (2, w["readmissions"], 0)) + (w["index"],),
    # The interleave rules order as auditions-first; the replay loop alternates the tiers (see replay()).
    "interleave+tokens": lambda w: _auditions_first(w, w["level0_tokens"]),
    "interleave": lambda w: _auditions_first(w, 0),
    "deep-first": lambda w: (not w["pending"],) + (
        (0, -w["levels"], w["readmissions"]) if w["levels"] >= 1
        else (1, 0, 0) if w["readmissions"] == 0 else (2, w["readmissions"], 0)) + (w["index"],),
}


def replay(games, report, rule, *, slots=None, marks=(3600.0, 7200.0)):
    """Replay phase 2 with a different backlog order: every game plays its own recorded phase-2 tenures in order
    (their durations, tokens and clears at their offsets), but a freed slot goes to the waiting game ``rule`` puts
    first. A game the rule admits after its recorded tenures ran out plays an unrecorded tenure (the median duration
    of its kind): that slot time is reported as unmodelled and valued at the recorded rate of its category (points per
    slot-second of level-0 or deep tenures) for a range. Static: the tenures do not change with their timing."""
    switch = report["scheduler"]["switch_wall_seconds"]
    wall_end = float(report["wall_seconds"])
    slots = slots or report["scheduler"].get("active_limit") or 23
    state, active, done = {}, [], []
    rates = {"level0": [0.0, 0.0], "deep": [0.0, 0.0]}  # points, slot-seconds of recorded tenures by kind
    for g in games:
        tenures = phase2_tenures(g, switch, wall_end)
        n = g.raw["total_levels"] or 8
        levels0 = g.level_at(switch + 1e-6)
        for t in tenures:
            t["kind"] = kind = "deep" if (g.level_at(t["start"] + 1e-6) >= 1) else "level0"
            rates[kind][0] += sum(points(n, lv) for _, lv in t["clears"])
            rates[kind][1] += t["duration"]
        kept = bool(tenures) and tenures[0]["reason_in"] == "opening_start"
        state[g.id] = {"game": g, "n": n, "tenures": tenures, "next": 0, "levels": levels0,
                       "admit_levels": levels0 if kept else 0,  # as the scheduler: the keep is an admission
                       "fruitless_on_level": 0,
                       "readmissions": 1 if kept else 0, "level0_tokens": g.tokens_at(switch) if levels0 == 0 else 0,
                       "pending": False, "index": g.index, "finished": False}
        if kept:
            active.append((switch + tenures[0]["duration"], g.id, switch, {**tenures[0], "cat": "deep"}, False))
            state[g.id]["next"] = 1
    medians = {kind: statistics.median([t["duration"] for s in state.values() for t in s["tenures"]
                                         if t["kind"] == kind]) for kind in ("level0", "deep")}
    rate = {k: (v[0] / v[1] if v[1] else 0.0) for k, v in rates.items()}
    clears, unmodelled = [], {"level0": 0.0, "deep": 0.0}
    order = RULES[rule]
    first_admission = {}
    now = switch

    deep_waits = []
    admissions = []
    last_pick = [None]
    categories = {c: {"slot_seconds": {int(m): 0.0 for m in marks}, "points": {int(m): 0.0 for m in marks}}
                  for c in ("audition", "reaudition", "deep")}

    def admit(sid, t):
        s = state[sid]
        cat = "deep" if s["levels"] >= 1 else "audition" if s["readmissions"] == 0 else "reaudition"
        admissions.append((round(t, 1), sid, s["levels"], s["readmissions"], s["fruitless_on_level"], cat))
        s["admit_levels"] = s["levels"]
        s["readmissions"] += 1
        s["pending"] = False
        first_admission.setdefault(sid, t)
        if s.get("parked_deep") is not None:
            deep_waits.append((sid, round(t - s["parked_deep"], 1)))
            s["parked_deep"] = None
        if s["next"] < len(s["tenures"]):
            tenure = s["tenures"][s["next"]]
            s["next"] += 1
            active.append((t + tenure["duration"], sid, t, {**tenure, "cat": cat}, False))
        else:
            kind = "deep" if s["levels"] >= 1 else "level0"
            active.append((t + medians[kind], sid, t, {"duration": medians[kind], "clears": [], "tokens": 0,
                                                       "end": "unrecorded", "kind": kind, "cat": cat}, True))

    while True:
        if now < wall_end:
            waiting = [sid for sid, s in state.items()
                       if not s["finished"] and all(a[1] != sid for a in active)]
            while len(active) < slots and waiting:
                best = min(waiting, key=lambda sid: order(state[sid]))
                if rule.startswith("interleave"):
                    # Alternate: after a first audition, the best waiting game with a level (if any) goes next.
                    deep = [sid for sid in waiting if state[sid]["levels"] >= 1 or state[sid]["pending"]]
                    if deep and last_pick[0] == "audition":
                        best = min(deep, key=lambda sid: order(state[sid]))
                    s_best = state[best]
                    last_pick[0] = ("audition" if s_best["levels"] == 0 and s_best["readmissions"] == 0
                                    and not s_best["pending"] else "other")
                waiting.remove(best)
                admit(best, now)
        if not active:
            break  # every tenure running at the wall is settled up to the wall
        active.sort()
        end, sid, start, tenure, extra = active.pop(0)
        s = state[sid]
        stop = min(end, wall_end)
        if extra:
            unmodelled[tenure["kind"]] += stop - start
        row = categories[tenure["cat"]]
        for mark in marks:
            row["slot_seconds"][int(mark)] += max(0.0, min(stop, mark) - start)
            row["points"][int(mark)] += sum(points(s["n"], s["levels"] + i + 1)
                                            for i, (offset, _) in enumerate(tenure["clears"]) if start + offset <= mark)
        for offset, _ in tenure["clears"]:
            if start + offset <= wall_end:
                s["levels"] += 1
                clears.append((start + offset, sid, s["levels"]))
        if s["levels"] == 0:
            s["level0_tokens"] += tenure["tokens"]
        if tenure["end"] == "finish" or (s["levels"] >= s["n"]):
            s["finished"] = True
        elif s["levels"] >= 1 and stop < wall_end:
            s["parked_deep"] = stop
        if s["levels"] == s["admit_levels"] and stop < wall_end:
            s["fruitless_on_level"] += 1  # a tenure without a clear
        elif s["levels"] > s["admit_levels"]:
            s["fruitless_on_level"] = 0
        now = max(now, stop)
    # Opening clears (before the switch) as recorded.
    opening = [(c, g.id, i + 1) for g in games for i, c in enumerate(g.clear_walls) if c <= switch]
    result = {"rule": rule, "slots": slots}
    for mark in marks:
        got = [(t, sid, lv) for t, sid, lv in opening + clears if t <= mark]
        score = sum(points(state[sid]["n"], lv) for _, sid, lv in got)
        result[f"points_{int(mark)}"] = round(score, 3)
        result[f"levels_{int(mark)}"] = len(got)
    result["unmodelled_slot_seconds"] = {k: round(v) for k, v in unmodelled.items()}
    result["unmodelled_points_at_recorded_rate"] = round(sum(unmodelled[k] * rate[k] for k in unmodelled), 3)
    result["first_admission"] = {sid: round(t) for sid, t in first_admission.items()}
    waits = [w for _, w in deep_waits]
    result["deep_readmission_waits"] = {
        "count": len(waits), "mean": round(statistics.mean(waits), 1) if waits else None,
        "median": statistics.median(waits) if waits else None, "over_300": sum(w > 300 for w in waits),
        "k035": [w for sid, w in deep_waits if sid == "k035"], "k018": [w for sid, w in deep_waits if sid == "k018"]}
    result["admissions"] = admissions
    result["categories"] = {c: {k: {m: round(v, 3) for m, v in d.items()} for k, d in row.items()}
                            for c, row in categories.items()}
    return result


def first_order(result, reference):
    """Slot time by category (first auditions, level-0 re-auditions, deep tenures) valued at the reference replay's
    recorded rates (points per slot-second of that category): symmetric in which games moved, unlike the clear
    accounting, which only credits recorded clears (a game moved later loses them; one moved earlier gains
    unmodelled time)."""
    out = {}
    for mark in ("3600", "7200"):
        value = 0.0
        for cat, row in result["categories"].items():
            ref = reference["categories"][cat]
            rate = ref["points"][int(mark)] / ref["slot_seconds"][int(mark)] if ref["slot_seconds"][int(mark)] else 0.0
            value += row["slot_seconds"][int(mark)] * rate
        out[mark] = round(value, 3)
    return out


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("trial_dir")
    ap.add_argument("--config", required=True)
    ap.add_argument("--out", default=None)
    args = ap.parse_args(argv)
    report, games = load(args.trial_dir, args.config)
    switch = report["scheduler"]["switch_wall_seconds"]
    wall_end = float(report["wall_seconds"])
    result = {"switch_wall_seconds": switch, "level0": level0(games, switch, wall_end), "parks": parks(games, wall_end),
              "replay": {rule: replay(games, report, rule) for rule in RULES}}
    recorded = sum(points(g.raw["total_levels"] or 8, i + 1) for g in games for i, c in enumerate(g.clear_walls))
    recorded_1h = sum(points(g.raw["total_levels"] or 8, i + 1) for g in games for i, c in enumerate(g.clear_walls)
                      if c <= 3600)
    print(f"recorded: {recorded:.3f} points at 2 h, {recorded_1h:.3f} at 1 h (level weights)")
    reference = result["replay"]["current"]
    print("| rule | clears: points 1 h / 2 h | unmodelled slot-s (level 0 / deep) | first order 1 h / 2 h | "
          "slot-s by 2 h: auditions / re-auditions / deep |")
    print("| --- | ---: | --- | ---: | ---: |")
    for rule, row in result["replay"].items():
        u = row["unmodelled_slot_seconds"]
        row["first_order"] = first_order(row, reference)
        c = row["categories"]
        print(f"| {rule} | {row['points_3600']} / {row['points_7200']} | {u['level0']} / {u['deep']} | "
              f"{row['first_order']['3600']} / {row['first_order']['7200']} | "
              f"{c['audition']['slot_seconds'][7200]:.0f} / {c['reaudition']['slot_seconds'][7200]:.0f} / "
              f"{c['deep']['slot_seconds'][7200]:.0f} |")
    rates = {cat: {m: round(1e4 * reference["categories"][cat]["points"][m]
                            / max(1.0, reference["categories"][cat]["slot_seconds"][m]), 4) for m in (3600, 7200)}
             for cat in reference["categories"]}
    print("recorded rates (points per 10K slot-seconds):", rates)
    print("deep re-admission waits:", {rule: row["deep_readmission_waits"] for rule, row in result["replay"].items()})
    for rule in ("current", "progress-fruitless+tokens"):
        rows = result["replay"][rule]["admissions"]
        second = [r for r in rows if r[5] == "reaudition" and 4900 <= r[0] <= 7200]
        mine = {k: [r[0] for r in rows if r[1] == k and r[0] >= 3400] for k in ("k018", "k021", "k014", "k016")}
        print(f"{rule}: second-round auditions 4,900-7,200 s: {len(second)} (first at "
              f"{second[0][0] if second else None}); re-admissions after 3,400 s: {mine}")
    watch = ("k095", "k075", "k067", "k105", "k035")
    print("first phase-2 admission:", {rule: {k: row["first_admission"].get(k) for k in watch}
                                       for rule, row in result["replay"].items()})
    lv = result["level0"]
    print(f"level-0 games parked at the switch: {lv['games']}; cleared L1 in phase 2: {lv['cleared']}; "
          f"quick (<= {QUICK_TOKENS} tokens after re-admission): {lv['quick']}")
    print("| order | AUC quick (95%) | AUC quick within family | AUC cleared | AUC cleared within family | "
          "mean position: quick / cleared |")
    print("| --- | ---: | ---: | ---: | ---: | ---: |")
    for name, o in lv["orders"].items():
        print(f"| {name} | {o['auc_quick']} {o['auc_quick_95']} | {o['auc_quick_within_family']} | {o['auc_cleared']} | "
              f"{o['auc_cleared_within_family']} ({o['families_with_both']}) | {o['mean_position_quick']} / "
              f"{o['mean_position_cleared']} |")
    print("\nlevel-budget parks at L >= 1:")
    print("| game | L | wall | parked s | no-op | since novel | new in last 20 | tokens since novel | "
          "tokens to next clear |")
    print("| --- | ---: | ---: | ---: | --- | ---: | ---: | ---: | ---: |")
    for r in sorted(result["parks"], key=lambda r: r["wall"]):
        e = r["evidence"]
        print(f"| {r['game']} {r['family']} | {r['levels']} | {r['wall']:.0f} | {r['parked_seconds']} | "
              f"{'yes' if r['noop'] else ''} | {e['since_novel']} | {e['new_last20']} | {e['tokens_since_novel']} | "
              f"{r['tokens_readmission_to_clear']} |")
    if args.out:
        Path(args.out).write_text(json.dumps(result, indent=1) + "\n")


if __name__ == "__main__":
    main()
