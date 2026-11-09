"""RL redesign data layer: roster split, harvested-start backlog, curriculum
ranking and level budgets. Behind ``mg.redesign``; pure Python, no I/O except ``extract_run`` / ``load_history``.

- Roster: the 15 easiest of the 20 c100/c110 games train (the 3-run v8.5p mean), the other 10 are held out.
- Backlog: a phase opens with ``streams // group_size`` groups. A try that reaches level k+1 saves that start (state
  + context) and its stream parks; with ``group_size`` streams parked the best start gets a group of ``group_size``
  tries (admitted streams stay within ``group_size`` of the cap). A saved start is used once and retires when the policy is ``retire_age`` updates newer than
  the version that saved it; L1 starts never leave. The phase ends after ``phase_groups`` full groups; groups still
  in flight carry over (their tries pause over the train step and go on after it, arena.phase) and count in the
  next phase when they fill; with ``carry_inflight`` off they are dropped and their starts go back unused.
- Ranking: population SD of per-try RHAE/tok over the freshest group played on the level (before any: every
  historical v8.5p try on it) x the level index. RHAE = level_score / 100 on a clear, 0 on a fail.
- Budgets: ``budget_mult`` x p80 of clear tokens; historical until fresh groups exist (1 for levels with >= 3
  historical clears, 2 below), then the clears of the last ``fresh_window`` groups. A level without historical clears
  takes the game's median level budget x the level index. Every budget is capped at ``budget_ceiling`` tokens.

Historical rows (``extract_run``): {"run", "instance", "game", "level", "cleared", "tokens", "actions", "rhae",
"censored"}; tokens are completion tokens on the level, actions env steps on it (as the level bench counts them), the
level a whole-game try was on at the wall end is ``censored``. A censored row counts as a fail only once its tokens
reach the level's historical budget (a budget-out); shorter ones are dropped.
"""

from __future__ import annotations

import json
import math
import random
import statistics
from collections import defaultdict
from pathlib import Path

from .level_points import level_score

# the 15 easiest of the 20 c100/c110 games train; the 5 hardest + the 5 left out of
# c100 are held out
TRAIN = ("tr87", "sb26", "m0r0", "r11l", "re86", "cn04", "sc25", "cd82", "tn36", "ka59", "dc22", "ls20", "su15",
         "g50t", "s5i5")
HELD_OUT = ("sp80", "wa30", "sk48", "lf52", "bp35", "ft09", "vc33", "lp85", "tu93", "ar25")
LEFT_OUT = ("ar25", "ft09", "lp85", "tu93", "vc33")

DEFAULTS = {
    "group_size": 12,
    "phase_groups": 16,
    "retire_age": 2,  # live: a saved start retires at policy version >= its version + retire_age
    "budget_mult": 1.25,
    "budget_q": 0.8,
    "min_hist_clears": 3,  # below: flagged; fresh budgets need 2 groups instead of 1
    "fresh_window": 2,
    "budget_ceiling": 300000,  # tokens per attempt, every budget (fallback included); None = none
    "unknown_first": True,  # a level with no tries at all (historical or fresh) ranks above every scored level
    # live, off by default (not in the spec): at most this many groups in flight per (game, level) for saved starts
    # while another start is free; an L1 start always takes one group at a time while another start is free
    "level_inflight": None,
    # at most this many saved starts per (game, level) (2); a new one replaces the oldest (None: no cap)
    "level_starts": None,
    # update mix (explicit approval pending; live, off by default). game_cap: at most this many groups of
    # one game per update, taken in finish order; a game's later groups wait for the next update, and a game holding
    # 2 x game_cap groups (in flight + full) launches no more while another game's start is free. min_update_tokens:
    # the update also waits until its groups hold this many tokens, up to max_update_groups (None: 2 x phase_groups).
    # v17 U21-U23: after a restart sb26 filled 8-11 of 16 groups at 0.5-0.8M tokens per update (U19/U20: ~5M).
    "game_cap": None,
    "min_update_tokens": None,
    "max_update_groups": None,
    "carry_inflight": True,  # a phase stop pauses in-flight tries; they finish after the train step
    # G-long (27 Oct, generation-only reads): every group's token budget (None: the curriculum's), and recorded starts
    # [{"game", "level", "attempt", "request"}] that stay after use (restored as harvested starts are)
    "budget_override": None,
    # floor every budget at the level's v8.5p tokens-to-clear quantile,
    # p75 when the budget came from fresh clears, p90 on the historical rule (no fresh clears yet). U8/U9: fresh
    # budgets near v8.5p's median cut whole groups (cd82-L3 0/8 at 34.4K vs v8.5p p50 38.5K)
    "budget_floor": False,
    # a new fresh budget never cuts more than 20% of the window's tries: at least the window's
    # p80 try (nearest rank) with every cut try valued at the budget it was cut at; >20% cut holds the budget there
    "budget_guard": False,
    # RL 29 Oct (live, off): budget-cut tries count at their budget in the p80 (a censored estimate: the rule's fixed
    # point without it cuts ~2x the intended share), so a budget can also rise; at most budget_mult x the latest budget
    "budget_censored": False,
    "seed_starts": [],
    # ranking: "rel" = rel_spread x level; "abs" = pstdev(RHAE/tok) x level (26 Oct spec)
    "rank": "rel",
    "rank_floor_frac": 0.5,
    "rank_min_n": 5,  # fewer tries in the scoring group (or history) rank behind every ranked level
    "seed": 0,  # sibling starts of one (game, level) tie-break by a seeded draw (RNG state in the backlog state)
}


def settings(mg):
    """``mg.redesign``: true or {"backlog": {...}} -> the backlog settings, None when the flag is off."""
    r = (mg or {}).get("redesign")
    if not r:
        return None
    cfg = (r.get("backlog") or {}) if isinstance(r, dict) else {}
    unknown = set(cfg) - set(DEFAULTS)
    if unknown:
        raise ValueError(f"mg.redesign.backlog: unknown keys {sorted(unknown)}")
    return {**DEFAULTS, **cfg}


def game_split(scores, left_out=LEFT_OUT, n_train=15):
    """{"train", "held_out"} from per-game scores of the roster games (highest train; ties by name)."""
    order = sorted((g for g in scores if g not in left_out), key=lambda g: (-scores[g], g))
    return {"train": order[:n_train], "held_out": order[n_train:] + [g for g in left_out]}


# ---- per-try metric ----

def rhae_tok(t):
    """Per-try RHAE / tokens on the level: RHAE (level_score / 100) on a clear, 0 on a fail."""
    if not t.get("cleared") or not t.get("tokens"):
        return 0.0
    return float(t["rhae"]) / float(t["tokens"])


def spread(tries):
    vals = [rhae_tok(t) for t in tries]
    return statistics.pstdev(vals) if len(vals) > 1 else 0.0


def rel_spread(tries, floor_frac=0.5):
    """pstdev(RHAE/tok) / median(RHAE/tok) of the same tries (scale-free: pstdev alone ranks
    levels cleared in few tokens first). A fail's RHAE/tok is 0, so a group with half or more fails has median 0:
    the denominator is floored at ``floor_frac`` x the mean (0 only when every try failed: no spread either)."""
    vals = [rhae_tok(t) for t in tries]
    if len(vals) < 2:
        return 0.0
    den = max(statistics.median(vals), floor_frac * statistics.fmean(vals))
    return statistics.pstdev(vals) / den if den > 0 else 0.0


def quantile(xs, q):
    """Linear interpolation between order statistics (numpy's default)."""
    xs = sorted(xs)
    if not xs:
        return None
    pos = (len(xs) - 1) * q
    lo = math.floor(pos)
    hi = min(lo + 1, len(xs) - 1)
    return xs[lo] + (xs[hi] - xs[lo]) * (pos - lo)


# ---- historical data ----

def extract_run(run_dir, baselines, run=None):
    """Per-level rows of one whole-game arena run (study.json + public-roster.json + games/<id>/log.jsonl)."""
    run_dir = Path(run_dir)
    study = json.loads((run_dir / "study.json").read_text())
    roster = {r["game_id"]: r["source_game_id"][:4] for r in json.loads((run_dir / "public-roster.json").read_text())}
    report = study["trials"][0]["report"]
    out, scores = [], defaultdict(list)
    for g in report["games"]:
        game = roster[g["game_id"]]
        scores[game].append(float(g["score"]))
        first = {}
        log = run_dir / "games" / g["game_id"] / "log.jsonl"
        if not log.is_file():  # a game whose log never reached the bucket: its score counts, no level rows
            continue
        with open(log) as fh:
            for line in fh:
                if '"observation"' not in line[:40]:
                    continue
                r = json.loads(line)
                if r.get("kind") == "observation":
                    first.setdefault(int(r["levels"]), int(r["step"]))
        cleared = int(g["levels"])
        first.setdefault(cleared, int(g["actions"]))  # the last clear's observation can miss the log at the wall
        cum = {0: 0, **{int(x["level"]): int(x["budget_seconds"]) for x in g.get("time_to_level") or []}}
        base = baselines.get(game) or []
        for k in range(1, cleared + 1):
            if k not in first or k - 1 not in first or k not in cum or k - 1 not in cum:
                continue  # a level whose first board or clear time the log missed: no row (short runs, 2-level jumps)
            actions = first[k] - first[k - 1]
            b = base[k - 1] if k <= len(base) else None
            out.append({"run": run, "instance": g["game_id"], "game": game, "level": k, "cleared": True,
                        "tokens": cum[k] - cum[k - 1], "actions": actions,
                        "rhae": round(level_score(b, actions, True) / 100.0, 6), "censored": False})
        if g.get("status") != "won" and cleared < int(g["total_levels"]) and cleared in first and cleared in cum:
            out.append({"run": run, "instance": g["game_id"], "game": game, "level": cleared + 1, "cleared": False,
                        "tokens": int(g["completion_tokens"]) - cum[cleared],
                        "actions": int(g["actions"]) - first[cleared], "rhae": 0.0, "censored": True})
    return {"rows": out, "scores": {k: round(statistics.mean(v), 4) for k, v in scores.items()}}


def load_baselines(points_path):
    """{game prefix: per-level human baseline actions} from a level-bench start-points file."""
    data = json.loads(Path(points_path).read_text())
    out = {}
    for p in data["points"]:
        out.setdefault(p["game"][:4], p.get("baselines") or [])
    return out


def load_history(path):
    """Rows saved by ``python -m arena.rl_backlog extract`` (a JSON {"rows": [...], ...})."""
    return json.loads(Path(path).read_text())["rows"]


# ---- curriculum: ranking and budgets ----

class Curriculum:
    def __init__(self, history_rows, cfg=None):
        self.cfg = {**DEFAULTS, **(cfg or {})}
        self.hist = defaultdict(list)
        for r in history_rows:
            self.hist[(r["game"], int(r["level"]))].append(r)
        self.fresh = defaultdict(list)  # (game, level) -> [groups (lists of tries)], oldest first, last fresh_window

    def record_group(self, game, level, tries):
        key = (game, int(level))
        self.fresh[key] = (self.fresh[key] + [[dict(t) for t in tries]])[-max(2, int(self.cfg["fresh_window"])):]

    def hist_clears(self, game, level):
        return [int(r["tokens"]) for r in self.hist.get((game, int(level)), []) if r["cleared"] and r["tokens"]]

    def _p(self, toks):
        return self.cfg["budget_mult"] * quantile(toks, self.cfg["budget_q"])

    def hist_budget(self, game, level):
        """1.25 x p80 of the level's historical clears; without clears the game's median level budget x level."""
        toks = self.hist_clears(game, level)
        if toks:
            return self._p(toks)
        cleared = {(g, k): c for (g, k) in self.hist if (c := self.hist_clears(g, k))}
        own = [self._p(c) for (g, _k), c in cleared.items() if g == game]
        if not own:  # a game without any historical clear: every game's levels
            own = [self._p(c) for c in cleared.values()]
        return statistics.median(own) * int(level) if own else None

    def budget(self, game, level):
        key = (game, int(level))
        need = 1 if len(self.hist_clears(game, level)) >= self.cfg["min_hist_clears"] else 2
        groups = self.fresh.get(key) or []
        b = None
        if len(groups) >= need:
            toks = [int(t["tokens"]) for grp in groups[-int(self.cfg["fresh_window"]):] for t in grp
                    if t.get("cleared") and t.get("tokens")]
            if toks:
                b = self._p(toks)
        fresh = b is not None
        if b is None:  # no fresh clears yet: the historical rule
            b = self.hist_budget(game, level)
        if self.cfg.get("budget_censored") and len(groups) >= need:
            win = groups[-int(self.cfg["fresh_window"]):]
            vals = self.valued(win)
            last = [float(t["budget"]) for t in win[-1] if t.get("budget")]
            if vals and last:
                c = min(self._p(vals), self.cfg["budget_mult"] * max(last))
                if b is None or c > b:
                    b = c
        if self.cfg.get("budget_guard") and len(groups) >= need:
            g = self.guard(groups[-int(self.cfg["fresh_window"]):])
            if g is not None and (b is None or g > b):
                b = g
        floor = self.floor(game, level, fresh)
        if floor is not None and (b is None or b < floor):
            b = floor
        b = self._cap(b)
        return None if b is None else int(math.ceil(b))

    @staticmethod
    def guard(groups, cut_at=0.95, share=0.8):
        """``budget_guard``: the window's p80 try by nearest rank (the ceil(0.8 n)-th smallest), a cut
        try (not cleared, tokens >= 0.95 x its group's budget) valued at that budget, so no more than 20% of the tries
        sit above it; > 20% cut at b gives b (the budget holds). Tries the hook never saw (0 tokens) are left out;
        tries recorded before budgets were kept count at their tokens. None without tries."""
        vals = sorted(Curriculum.valued(groups, cut_at))
        return vals[max(0, math.ceil(share * len(vals)) - 1)] if vals else None

    @staticmethod
    def valued(groups, cut_at=0.95):
        """Every seen try's tokens, a cut try (not cleared, >= 0.95 x its group's budget) at that budget."""
        vals = []
        for grp in groups:
            for t in grp:
                tokens = int(t.get("tokens") or 0)
                if tokens <= 0:
                    continue
                b = t.get("budget")
                cut = not t.get("cleared") and b and tokens >= cut_at * float(b)
                vals.append(float(b) if cut else float(tokens))
        return vals

    def floor(self, game, level, fresh):
        """``budget_floor`` on: the level's v8.5p p75 (fresh) / p90 (historical rule) tokens-to-clear, else None."""
        if not self.cfg.get("budget_floor"):
            return None
        return quantile(self.hist_clears(game, level), 0.75 if fresh else 0.9)

    def _cap(self, b):
        ceiling = self.cfg.get("budget_ceiling")
        return min(b, float(ceiling)) if b is not None and ceiling else b

    def hist_tries(self, game, level):
        """Historical tries of the level; a censored try counts as a fail only past the level's historical budget."""
        rows = self.hist.get((game, int(level)), [])
        cut = None
        out = []
        for r in rows:
            if r.get("censored"):
                cut = self._cap(self.hist_budget(game, level)) if cut is None else cut
                if cut is None or r["tokens"] < cut:
                    continue
            out.append(r)
        return out

    def source(self, game, level):
        groups = self.fresh.get((game, int(level)))
        if groups:
            return "fresh", groups[-1]
        tries = self.hist_tries(game, level)
        return ("hist", tries) if tries else (None, [])

    def score(self, game, level):
        """spread x level index over the freshest group (the v8.5p history before any); None when the level has no
        tries at all. rank "rel": pstdev / median of RHAE/tok (rel_spread); "abs": pstdev of RHAE/tok."""
        src, tries = self.source(game, level)
        if src is None:
            return None
        if self.cfg.get("rank", "rel") == "abs":
            return spread(tries) * int(level)
        return rel_spread(tries, float(self.cfg.get("rank_floor_frac", 0.5))) * int(level)

    def rank_key(self, game, level):
        """Larger ranks first: unscored levels on top with ``unknown_first``, else at the bottom; a level scored on
        fewer than ``rank_min_n`` tries behind every ranked level (rank "rel" only; ties: level, then the draw)."""
        s = self.score(game, level)
        if s is None:
            return math.inf if self.cfg["unknown_first"] else -math.inf
        if self.cfg.get("rank", "rel") != "abs" and len(self.source(game, level)[1]) < int(self.cfg.get("rank_min_n") or 0):
            return -math.inf
        return s

    def table(self, games):
        rows = []
        levels = sorted({k for (g, k) in set(self.hist) | set(self.fresh) if g in games})
        for game in games:
            for k in levels:
                if (game, k) not in self.hist and (game, k) not in self.fresh:
                    continue
                toks = self.hist_clears(game, k)
                src, tries = self.source(game, k)
                rows.append({"game": game, "level": k, "hist_tries": len(self.hist.get((game, k), [])),
                             "hist_clears": len(toks), "p80": quantile(toks, self.cfg["budget_q"]),
                             "budget": self.budget(game, k), "source": src, "tries": len(tries),
                             "spread": spread(tries) if tries else None, "score": self.score(game, k),
                             "flag": len(toks) < self.cfg["min_hist_clears"]})
        return rows

    def state_dict(self):
        return {"fresh": [[g, k, grps] for (g, k), grps in sorted(self.fresh.items())]}

    def load_state(self, state):
        self.fresh = defaultdict(list, {(g, int(k)): grps for g, k, grps in state.get("fresh") or []})


def _num(v, scale=1.0):
    return "-" if v is None else f"{v * scale:.3f}"


def format_table(rows):
    head = f"{'game':5} {'L':>2} {'h.tries':>7} {'h.clr':>5} {'p80':>8} {'budget':>8} {'src':>5} {'n':>3} " \
           f"{'spread/1e4':>10} {'score/1e4':>9}  flag"
    lines = [head]
    for r in rows:
        lines.append(f"{r['game']:5} {r['level']:>2} {r['hist_tries']:>7} {r['hist_clears']:>5} "
                     f"{'-' if r['p80'] is None else int(r['p80']):>8} {'-' if r['budget'] is None else r['budget']:>8} "
                     f"{r['source'] or '-':>5} {r['tries']:>3} {_num(r['spread'], 1e4):>10} {_num(r['score'], 1e4):>9}  "
                     f"{'<3 clears' if r['flag'] else ''}")
    return "\n".join(lines)


# ---- harvested-start backlog ----

class Backlog:
    """Phase controller API: ``start_phase`` -> groups; per try ``on_level_reached`` or ``on_try_done``; poll
    ``next_group`` after each; ``end_phase`` once ``phase_done``; ``on_update(version)`` after the train step.

    A group is {"id", "start": start row, "game", "level", "tries", "budget", "version"}; a start row is {"id",
    "game", "level", "state", "context", "version", "l1"} (state and context opaque, JSON-serialisable)."""

    def __init__(self, games, curriculum, streams, cfg=None, version=0):
        self.cfg = {**DEFAULTS, **(cfg or {})}
        self.cur = curriculum
        self.streams = int(streams)
        self.version = int(version)
        self.starts = {}
        for g in games:
            sid = f"{g}-L1"
            self.starts[sid] = {"id": sid, "game": g, "level": 1, "state": None, "context": None, "version": None,
                                "l1": True}
        self.groups = {}  # in flight: id -> {"group", "results": {try index: try row}}
        self.full = []  # this phase's full groups: {"group", "tries"}
        self.parked = 0
        self.phase = 0
        self.counter = 0
        self.stats = {"saved": 0, "used": 0, "retired": 0, "returned": 0}
        self.rng = random.Random(int(self.cfg.get("seed") or 0))
        for i, seed in enumerate(self.cfg.get("seed_starts") or []):
            if seed["game"] not in games:
                continue
            sid = f"{seed['game']}-L{int(seed['level'])}-seed{i}"
            self.starts[sid] = {"id": sid, "game": seed["game"], "level": int(seed["level"]),
                                "state": {"attempt": seed["attempt"], "request": int(seed["request"])}, "context": None,
                                "version": 0, "l1": False, "keep": True}
        for st in self.starts.values():
            st["tie"] = self.rng.random()

    # -- selection --
    def _available(self):
        inflight, per_level = defaultdict(int), defaultdict(int)
        for g in self.groups.values():
            inflight[g["group"]["start"]["id"]] += 1
            per_level[(g["group"]["game"], g["group"]["level"])] += 1
        cap = self.cfg.get("level_inflight")
        free = [s for s in self.starts.values() if not (s["l1"] and inflight[s["id"]])
                and (s["l1"] or cap is None or per_level[(s["game"], s["level"])] < int(cap))]
        held = self.held_games()
        if held and [s for s in free if s["game"] not in held]:
            free = [s for s in free if s["game"] not in held]
        if free:
            return free
        # every start is an L1 already in flight: the least-busy L1 again rather than idle streams
        # (only L1s: a saved start held back by level_inflight stays held back; 28 Oct: every free start came back)
        l1 = [s for s in self.starts.values() if s["l1"]]
        low = min((inflight[s["id"]] for s in l1), default=0)
        return [s for s in l1 if inflight[s["id"]] == low]

    def _key(self, s):
        """Rank, then level, then the start's seeded random draw (random among sibling starts). A start a
        resume returned (its group's tries died with the container) reopens before any other (v17 a8, 30 Oct)."""
        return (bool(s.get("resumed")), self.cur.rank_key(s["game"], s["level"]), s["level"], s["tie"], s["id"])

    def _launch(self):
        cands = self._available()
        if not cands:
            return None
        s = max(cands, key=self._key)
        s.pop("resumed", None)
        if not s["l1"] and not s.get("keep"):
            del self.starts[s["id"]]
            self.stats["used"] += 1
        self.counter += 1
        n = int(self.cfg["group_size"])
        grp = {"id": f"p{self.phase}g{self.counter}", "start": s, "game": s["game"], "level": s["level"], "tries": n,
               "budget": int(self.cfg.get("budget_override") or self.cur.budget(s["game"], s["level"])),
               "version": self.version}
        self.groups[grp["id"]] = {"group": grp, "results": {}}
        self.parked -= n
        return grp

    # -- phase --
    def start_phase(self):
        """All streams free: as many groups as fit (streams // group_size), best-ranked first."""
        self.phase += 1
        n = int(self.cfg["group_size"])
        self.parked = self.streams - sum(g["group"]["tries"] - len(g["results"]) for g in self.groups.values())
        out = []
        while self.parked >= n:
            g = self._launch()
            if g is None:
                break
            out.append(g)
        return out

    def held_games(self):
        """game_cap's launch hold: games with 2 x game_cap groups not yet trained (in flight + full)."""
        cap = self.cfg.get("game_cap")
        if not cap:
            return set()
        n = defaultdict(int)
        for g in self.groups.values():
            n[g["group"]["game"]] += 1
        for fg in self.full:
            n[fg["group"]["game"]] += 1
        return {g for g, k in n.items() if k >= 2 * int(cap)}

    def update_pick(self):
        """(indices into ``full`` this update trains, done): finish order; game_cap skips a capped game's later groups;
        done at phase_groups picked groups holding min_update_tokens, or at max_update_groups picked."""
        k = int(self.cfg["phase_groups"])
        cap, need = self.cfg.get("game_cap"), int(self.cfg.get("min_update_tokens") or 0)
        most = max(k, int(self.cfg.get("max_update_groups") or (2 * k if need else k)))
        per, picked, tokens = defaultdict(int), [], 0
        for i, fg in enumerate(self.full):
            game = fg["group"]["game"]
            if cap and per[game] >= int(cap):
                continue
            per[game] += 1
            picked.append(i)
            tokens += sum(int(t.get("tokens") or 0) for t in fg.get("tries") or [])
            if len(picked) >= most or (len(picked) >= k and tokens >= need):
                return picked, True
        return picked, False

    @property
    def phase_done(self):
        return self.update_pick()[1]

    def next_group(self):
        """A new group whenever group_size streams are free (admitted streams stay within
        [streams - group_size, streams]; a group never waits for another group's tries) and the phase is open."""
        if self.phase_done or self.parked < int(self.cfg["group_size"]):
            return None
        return self._launch()

    def on_level_reached(self, group_id, index, *, state, context, version, tokens, actions, rhae):
        """Try ``index`` of ``group_id`` cleared its level: save the next level's start, end the try (park)."""
        grp = self.groups[group_id]["group"]
        level = grp["level"] + 1
        sid = f"{grp['game']}-L{level}-{group_id}-t{index}"
        cap = self.cfg.get("level_starts")
        if cap is not None:  # the oldest saved starts of this level go (policy version, then save order)
            same = [s for s in self.starts.values() if not s["l1"] and s["game"] == grp["game"] and s["level"] == level]
            for s in sorted(same, key=lambda s: (s["version"], s.get("saved", 0)))[:max(0, len(same) - int(cap) + 1)]:
                del self.starts[s["id"]]
                self.stats["capped"] = self.stats.get("capped", 0) + 1
        self.saves = getattr(self, "saves", 0) + 1
        self.starts[sid] = {"saved": self.saves, "id": sid, "game": grp["game"], "level": level, "state": state, "context": context,
                            "version": int(version), "l1": False, "tie": self.rng.random()}
        self.stats["saved"] += 1
        self.on_try_done(group_id, index, cleared=True, tokens=tokens, actions=actions, rhae=rhae)
        return sid

    def on_try_done(self, group_id, index, *, cleared, tokens, actions=0, rhae=0.0):
        """A try ended (clear without a next level, fail, budget-out): its stream parks; a group's last try makes it
        full (recorded for the curriculum). Returns the full group or None; repeats are ignored."""
        live = self.groups.get(group_id)
        if live is None or index in live["results"]:
            return None
        live["results"][index] = {"cleared": bool(cleared), "tokens": int(tokens or 0), "actions": int(actions or 0),
                                  "rhae": float(rhae or 0.0) if cleared else 0.0,
                                  "budget": live["group"].get("budget")}  # budget_guard: cut tries count at it
        self.parked += 1
        grp = live["group"]
        if len(live["results"]) < grp["tries"]:
            return None
        del self.groups[group_id]
        tries = [live["results"][i] for i in sorted(live["results"])]
        self.cur.record_group(grp["game"], grp["level"], tries)
        done = {"group": grp, "tries": tries}
        self.full.append(done)  # past phase_groups: the next phase's
        return done

    def end_phase(self):
        """Close the phase: ``update_pick``'s full groups to train on (the first ``phase_groups`` with the update-mix
        knobs off; the rest open the next phase).
        Groups still in flight carry over, or with ``carry_inflight`` off are dropped and their starts go back."""
        picked, _ = self.update_pick()
        keep = set(picked)
        full, self.full = [self.full[i] for i in picked], [fg for i, fg in enumerate(self.full) if i not in keep]
        if self.cfg.get("carry_inflight", True):
            return {"full": full, "aborted": [], "carried": sorted(self.groups)}
        return {"full": full, "aborted": self.abandon_inflight(), "carried": []}

    def abandon_inflight(self):
        """Drop every in-flight group (a resume: their tries died with the container); saved starts go back unused,
        first in line and with their age clock restarted at the current version (v17 a8: 54 starts retired at once
        after a resume chain and the curriculum fell back to L1 starts)."""
        aborted = [g["group"] for g in self.groups.values()]
        for grp in aborted:
            s = grp["start"]
            if not s["l1"]:
                s.update(resumed=True, aged_from=self.version)
                self.starts[s["id"]] = s
                self.stats["returned"] += 1
                self.stats["used"] -= 1
        self.groups = {}
        self.parked = self.streams
        return [g["id"] for g in aborted]

    def drop_failed(self, group_id):
        """A group whose bench raised before any try finished (v17 30 Oct: a start folder deleted from the volume): out
        of flight with no curriculum record, its streams free, its start consumed (not returned: it would fail again)."""
        live = self.groups.pop(group_id, None)
        if live is None:
            return None
        self.parked += int(live["group"]["tries"]) - len(live["results"])
        self.stats["failed_starts"] = self.stats.get("failed_starts", 0) + 1
        return live["group"]

    def drop_full(self, gids):
        """Full groups a resume cannot train (their saved tries are missing): out of ``full``, their saved starts back
        unused, as ``abandon_inflight`` does (their curriculum record stays)."""
        gids = set(gids)
        for fg in [fg for fg in self.full if fg["group"]["id"] in gids]:
            s = fg["group"]["start"]
            if not s["l1"]:
                self.starts[s["id"]] = s
                self.stats["returned"] += 1
                self.stats["used"] -= 1
        self.full = [fg for fg in self.full if fg["group"]["id"] not in gids]

    def on_update(self, version, retire_age=None, played=True):
        """The policy is now ``version``: saved starts ``retire_age`` (live) or more updates old retire. A start ages
        only over updates that followed a play phase: ``played`` False (a resume's pending update, a replay) moves
        every saved start's age clock (``aged_from``, default its saving version) along with the version."""
        moved = int(version) - self.version
        self.version = int(version)
        if retire_age is not None:
            self.cfg["retire_age"] = int(retire_age)
        age = int(self.cfg["retire_age"])
        if not played and moved > 0:
            for s in self.starts.values():
                if not s["l1"]:
                    s["aged_from"] = int(s.get("aged_from", s["version"])) + moved
        gone = [sid for sid, s in self.starts.items()
                if not s["l1"] and self.version - int(s.get("aged_from", s["version"])) >= age]
        for sid in gone:
            del self.starts[sid]
        self.stats["retired"] += len(gone)
        return gone

    def ranking(self):
        """Available starts best first (game, level, score) for logging."""
        return [(s["id"], s["game"], s["level"], self.cur.score(s["game"], s["level"]))
                for s in sorted(self._available(), key=self._key, reverse=True)]

    # -- state --
    def state_dict(self):
        return {"version": self.version, "phase": self.phase, "counter": self.counter, "parked": self.parked,
                "starts": list(self.starts.values()), "groups": list(self.groups.values()), "full": self.full,
                "stats": dict(self.stats), "cfg": dict(self.cfg), "curriculum": self.cur.state_dict(),
                "rng": _rng_state(self.rng)}

    def load_state(self, state):
        self.version, self.phase, self.counter = int(state["version"]), int(state["phase"]), int(state["counter"])
        self.parked = int(state["parked"])
        self.starts = {s["id"]: s for s in state["starts"]}
        self.groups = {g["group"]["id"]: {"group": g["group"], "results": {int(i): r for i, r in g["results"].items()}}
                       for g in state["groups"]}
        self.full = list(state["full"])
        self.stats = dict(state["stats"])
        self.cfg = {**self.cfg, **(state.get("cfg") or {})}
        self.cur.load_state(state.get("curriculum") or {})
        if state.get("rng"):
            self.rng.setstate((state["rng"][0], tuple(state["rng"][1]), state["rng"][2]))
        for st in list(self.starts.values()) + [g["group"]["start"] for g in self.groups.values()]:
            if "tie" not in st:  # a state saved before the draw (T4 leg 1): drawn now, from the restored RNG
                st["tie"] = self.rng.random()


def _rng_state(rng):
    version, internal, gauss = rng.getstate()
    return [version, list(internal), gauss]


def main(argv=None):
    """``extract RUN_DIR... --points P --out F`` saves historical rows; ``table F`` prints split, ranking, budgets."""
    import argparse

    ap = argparse.ArgumentParser(prog="python -m arena.rl_backlog")
    sub = ap.add_subparsers(dest="cmd", required=True)
    ex = sub.add_parser("extract")
    ex.add_argument("runs", nargs="+")
    ex.add_argument("--points", required=True)
    ex.add_argument("--out", required=True)
    tb = sub.add_parser("table")
    tb.add_argument("history")
    a = ap.parse_args(argv)
    if a.cmd == "extract":
        base = load_baselines(a.points)
        rows, scores = [], {}
        for run in a.runs:
            got = extract_run(run, base, run=Path(run).name)
            rows += got["rows"]
            scores[Path(run).name] = got["scores"]
        Path(a.out).write_text(json.dumps({"scores": scores, "rows": rows}, indent=None, separators=(",", ":")))
        print(f"{len(rows)} rows -> {a.out}")
        return
    data = json.loads(Path(a.history).read_text())
    for run, sc in (data.get("scores") or {}).items():
        print(f"v8.5p score per game ({run}):", ", ".join(f"{g} {s:.1f}" for g, s in sorted(sc.items(), key=lambda x: -x[1])))
    print("TRAIN", " ".join(TRAIN))
    print("HELD_OUT", " ".join(HELD_OUT))
    cur = Curriculum(data["rows"])
    rows = cur.table(list(TRAIN))
    print(format_table(rows))
    print("ranking (L1 + every historical level, best first):")
    rk = sorted((r for r in rows if r["score"] is not None), key=lambda r: -r["score"])
    print("  " + ", ".join(f"{r['game']}-L{r['level']} {r['score'] * 1e4:.3f}" for r in rk))
    print("flagged (< 3 historical clears):", " ".join(f"{r['game']}-L{r['level']}" for r in rows if r["flag"]))


if __name__ == "__main__":
    main()
