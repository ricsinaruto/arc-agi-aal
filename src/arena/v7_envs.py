"""Envs Run 1 data setup on the v7 async loop (hero run, docs/envs-plan.md section 10).

On when ``mg.async`` and ``mg.curriculum`` are both set. Pieces:

- **Start pool** (``mg.harvest``): starts cut by explorers, by training clears (``harvest.collect`` per finished
  group) and the seed screen's recordings. A start is drawn at most ``max_uses`` times and only while
  ``book.lag(played_by) <= stale`` (versions behind the trainer; a version a rollback or a lost leg dropped from the
  lineage is stale). Fewest uses first, then the newest version.
- **Curriculum per group**: ``Curriculum.plan_one`` (source by deficit against the shares, frontier score, family and
  near-solved caps over the last groups); units without a start now, and games not active, are excluded, so a group
  always goes where a start exists (level 1 when no deeper start is fresh). Each group plays at its unit's cap
  (running cap 1.25 x p80 of clears <= 262K, times the unit's tier when most failures ran out of tokens).
- **Explorers** (``Chains``): an explorer plays one game from level 1 as a chain of one-try segments, each on the
  version served when it starts (it holds that version in the book like a group). A clear's harvest request is cut as
  the next level's start: it joins the pool (when the game has a training set at that level) and the chain's next
  segment restores it. A segment that spends ``seg_tokens`` without clearing is cut at its last request and the next
  segment restores that request: the same recorded conversation, replayed and checked request by request, so a
  continued segment sends the prompt the unsegmented play would have sent. Public games play the whole budget;
  other sources stop when stuck (``stuck`` x the level's cap on that level).
- **Rotation** (``mg.rotation``): sources in ``keep`` (public) never rotate; another game leaves the active set once it
  stops teaching (every observed unit retired or parked, L1 observed, explored at least ``min_explored`` times, no
  unobserved unit with a start), and the next reserve game (sets marked ``reserve``) of the same source comes in.
- **Chains**: every cleared try's next-level start carries ``chain_tokens`` (its
  parent start's plus the clear's tokens); up to ``curriculum.chain_groups`` groups per step continue a game from such a
  start (``Curriculum.plan_chain``: one per game per step, family cap, a bonus on the game's deepest open level, a
  near-certain level at a reduced quota) while ``chain_tokens`` + the level's cap stays inside ``chain_budget``; a chain
  level with ``chain_stuck`` consecutive clear-less chain groups closes for ``chain_reopen`` steps (or until any group
  clears it).
- **Whole-game read** (``mg.wread``): every ``every`` steps (and the base at step 0) each held-out game
  plays from level 1 on that step's served version for ``budget`` tokens, as one try cut into ``seg_tokens`` segments
  (an explorer chain: a clear's next-level start restores, a budget cut restores its last request); scored with the
  competition formula (levels weighted by index, all levels of the game). Every level reached becomes a held-out start
  (read only, never trained); at whole-game read steps the per-level read adds up to ``extra_sets`` such levels
  (first start captured, then fixed), reported apart from the core sets.
- **State**: pool, uses, running caps, explorer chains, rotation and the first-deeper-start time ride in the v7 loop
  state (``envs``); a resume restores them and restarts each chain from its last start.
"""
from __future__ import annotations

import asyncio
import json
import os
import shutil
import time
from pathlib import Path
from types import SimpleNamespace

from . import harvest as harvest_mod
from .play_explorer import CAP_KEYS, Explorers, record_tokens, update_caps
from .play_explorer import snapshot as cut_rows

# stop teaching per unit: retired or parked by the curriculum, or (its own thresholds, so the curriculum's park and
# retire can stay off as in Run 1) at least min_tries this run cleared at >= retire_p or <= park_p
ROTATION_KEYS = {"keep": ["public"], "min_explored": 1, "every": 1, "retire_p": 0.95, "park_p": 0.10, "min_tries": 12}


WREAD_KEYS = {"every": 8, "budget": 300000, "seg_tokens": 65536, "min_seg": 2048, "wall": 28800, "extra_sets": 24,
              "base": True}


def wread_settings(mg):
    w = mg.get("wread")
    if not w:
        return None
    unknown = set(w) - set(WREAD_KEYS)
    if unknown:
        raise ValueError(f"mg.wread: unknown keys {sorted(unknown)}")
    out = {**WREAD_KEYS, **w}
    stub = os.environ.get("ARC_PREFLIGHT_STUB") or ""
    if stub.startswith("{") and json.loads(stub).get("shrink"):  # CPU preflight: a whole-game read every 2 steps,
        out.update(every=2, budget=32768, seg_tokens=16384, min_seg=1024)  # a few segments each
    return out


def game_score(cleared, total):
    """The competition's game score (arc_agi EnvironmentScoreCalculator, as taaf.game._compute_final_score): every
    level of the game weighted by its 1-based index, a cleared level's score (0-115) else 0, capped at the scored
    levels' weight share x 100. ``cleared``: {level: score}."""
    total = max(int(total or 0), max([int(k) for k in cleared] or [0]))
    if total <= 0:
        return 0.0
    weights = sum(range(1, total + 1))
    got = sum(float(sc) * int(lv) for lv, sc in cleared.items())
    scored = sum(int(lv) for lv, sc in cleared.items() if float(sc) > 0)
    return min(got / weights, 100.0 * scored / weights)


def mean_se(xs):
    xs = [float(x) for x in xs]
    if not xs:
        return None, None
    m = sum(xs) / len(xs)
    if len(xs) < 2:
        return m, None
    return m, (sum((x - m) ** 2 for x in xs) / (len(xs) - 1) / len(xs)) ** 0.5


def ratio_ci(v, se, v0, se0):
    """v / v0 with a 95% interval (delta method, independent reads), or None."""
    if not v0 or v is None:
        return None
    r = float(v) / float(v0)
    d = 1.96 * abs(r) * (((se or 0.0) / v) ** 2 + ((se0 or 0.0) / v0) ** 2) ** 0.5 if v else None
    return [round(r, 4), None if d is None else round(max(0.0, r - d), 4), None if d is None else round(r + d, 4)]


def rotation_settings(mg):
    r = mg.get("rotation")
    if not r:
        return None
    unknown = set(r) - set(ROTATION_KEYS)
    if unknown:
        raise ValueError(f"mg.rotation: unknown keys {sorted(unknown)}")
    return {**ROTATION_KEYS, **r}


def code_of(game):
    return str(game)[:4]


def cut_clear(attempt, dest, row, level, checks=None, game=None, ceiling=None):
    """A cleared segment's harvest request cut as the start of ``level`` (the next level): ``((dest, request,
    prompt), None)`` or ``(None, why)``. Checks as harvest.collect: the next level's first board, no unsafe event
    before it, the prompt ceiling, ``restore.load``, the CPU board check."""
    from .restore import load, unsafe_from

    mark = row.get("harvest")
    if not mark:
        return None, "no_harvest_request"
    if mark.get("levels") is not None and int(mark["levels"]) + 1 != int(level):
        return None, "level_mismatch"
    if mark.get("state") not in (None, "NOT_FINISHED"):
        return None, "game_state"
    prompt = int(mark.get("prompt_tokens") or 0)
    if ceiling and prompt > int(ceiling):
        return None, "prompt_over_ceiling"
    try:
        harvest_mod.snapshot(attempt, dest)
        request = int(mark["request"])
        first, _ = unsafe_from(dest)
        if first is not None and first <= int(mark.get("step") or 0):
            raise ValueError("unsafe_event")
        if load(dest, request)["level"] != int(level):
            raise ValueError("start_mismatch")
        if checks is not None:
            ok, detail = harvest_mod.board_check(*checks, game, dest, request)
            if not ok:
                raise ValueError("board_check")
    except (OSError, ValueError, KeyError, IndexError) as error:
        shutil.rmtree(dest, ignore_errors=True)
        why = str(error) if str(error) in ("unsafe_event", "start_mismatch", "board_check") else \
            f"snapshot_{type(error).__name__}"
        return None, why
    return (Path(dest), request, prompt), None


def cut_mid(attempt, dest, after):
    """A segment that ran out of its tokens mid-level, cut at its last recorded gameplay request (later than
    ``after``, the request it was restored at): ``((dest, request), None)`` or ``(None, why)``. Restoring it replays
    every earlier reply from the recording and sends that request again live."""
    from .restore import load, unsafe_from

    attempt = Path(attempt)
    try:
        rows = (attempt / "rollout.jsonl").read_bytes()
        rows = [json.loads(x) for x in rows[: rows.rfind(b"\n") + 1].splitlines()]
    except (OSError, ValueError):
        return None, "no_rollout"
    live = [int(r["n"]) for r in rows if r.get("gameplay")]
    if not live or max(live) <= int(after):
        return None, "no_progress"
    request = max(live)
    try:
        cut_rows(attempt, dest, request)
        info = load(dest, request)
        first, _ = unsafe_from(dest)
        if first is not None and first <= int(info["stop"]["steps"]):
            raise ValueError("unsafe_event")
    except (OSError, ValueError, KeyError, IndexError) as error:
        shutil.rmtree(dest, ignore_errors=True)
        return None, str(error) if str(error) == "unsafe_event" else f"cut_{type(error).__name__}"
    return (Path(dest), request), None


def admit_staggered(key="explore-chains", record=False):
    """Explorer chains share the bench tries' admission stagger (mg.admit_stagger_s on the shared pool, one rule:
    ): at most one new or resumed chain per stagger. ``record``: the admission happened."""
    from . import level_bench

    pool = level_bench.SHARED_SLOTS
    stagger = float(getattr(pool, "stagger_s", 0.0) or 0.0) if pool is not None else 0.0
    if stagger <= 0:
        return True
    if record:
        pool.last_take[key] = time.time()
        return True
    return time.time() - pool.last_take.get(key, -1e18) >= stagger


class Chains(Explorers):
    """Explorer chains on the v7 loop (see the module doc). ``mg.explore`` keys as play_explorer plus
    ``seg_tokens``."""

    def __init__(self, loop, restored=None):
        super().__init__(loop)
        self.pending = [dict(c) for c in (restored or {}).get("chains") or []]
        self.count = max(self.count, int((restored or {}).get("count") or 0))
        self.first_deep = (restored or {}).get("first_deep")
        self.t0 = float((restored or {}).get("t0") or time.time())
        self.seg_tokens = 0  # tokens of finished segments since the last step row
        self.done_chains = int((restored or {}).get("done") or 0)
        self.source = {code_of(x["game"]): x.get("source") for x in loop.cfg["mg_sets"].values()
                       if x.get("role") == "train"}  # every game, reserve ones included
        self.explored = set((restored or {}).get("explored") or [])  # order unplayed_first: games an explorer took

    # ---- state ----
    def state(self):
        chains = [{k: v for k, v in c.items() if k != "task"} for c in self.live.values()] + list(self.pending)
        return {"count": self.count, "chains": json.loads(json.dumps(chains)), "first_deep": self.first_deep,
                "t0": self.t0, "done": self.done_chains, "explored": sorted(self.explored)}

    def played(self):
        """Game codes with a training group in this lineage (any curriculum unit observed)."""
        curr = getattr(self.loop, "mg_curr", None)
        return {code_of(u.game) for u in (curr.units.values() if curr is not None else []) if u.seen or u.picked}

    def unplayed_first(self, busy):
        """order unplayed_first: a never-played active game, then a candidate (reserve) game, neither explored yet."""
        played, active = self.played(), self.loop.v7e_active()
        fresh = [c for c in sorted(active) if c not in played]
        extra = [c for c in self.cfg["candidates"] if c in self.source and c not in active]
        for code in fresh + extra:
            if code not in self.explored and busy.get(code, 0) < int(self.cfg["per_game"]):
                self.explored.add(code)
                return code
        return None

    # ---- scheduling ----
    def next_game(self):
        active = self.loop.v7e_active()
        for code in list(active):
            if code not in self.rot:
                self.rot.append(code)
        busy = {}
        for p in self.live.values():
            busy[p["code"]] = busy.get(p["code"], 0) + 1
        if self.cfg.get("order") == "unplayed_first":
            code = self.unplayed_first(busy)
            if code is not None:
                return code
        for _ in range(len(self.rot)):
            code = self.rot[0]
            self.rot.rotate(-1)
            if code in active and busy.get(code, 0) < int(self.cfg["per_game"]):
                return code
        return None

    async def release(self):  # versions are the loop's (book), never explorer-owned names
        return None

    async def collect_one(self, name, rec, final=False):  # starts are cut at each segment's end
        return []

    async def launch(self):
        while len(self.live) < self.limit() and not self.stopping:
            if not admit_staggered():
                return  # the next poll admits the next one
            if self.pending:
                ch = self.pending.pop(0)
                ch["resumed"] = int(ch.get("resumed") or 0) + 1
            else:
                code = self.next_game()
                if code is None:
                    return
                self.count += 1
                self.loop.mg_explore_count = self.count
                ch = {"name": f"explore-{self.count:04d}-{code}", "n": self.count, "code": code, "level": 1,
                      "start": None, "spent": 0, "level_spent": 0, "levels": 0, "segs": 0, "cleared": [],
                      "versions": [], "started": time.time()}
            self.live[ch["name"]] = ch
            admit_staggered(record=True)
            ch["task"] = asyncio.ensure_future(self.chain(ch))

    def level_cap(self, code, level):
        """The level's cap for the stuck rule: its running cap once it has clears (or a probe prior), else
        ``nodata_cap`` (as play_explorer.point)."""
        loop = self.loop
        obs = (loop.__dict__.get("mg_level_obs") or {}).get(f"{code}-L{int(level)}") or {}
        s = loop.v7e_level_set(code, level)
        if s is None:
            return int(self.cfg["nodata_cap"])
        has = int((s.get("prior") or {}).get("clears") or 0) > 0 or len(obs.get("clears") or []) >= CAP_KEYS["min_clears"]
        return int(s.get("cap") or self.cfg["nodata_cap"]) if has else int(self.cfg["nodata_cap"])

    async def chain(self, ch):
        loop, why = self.loop, None
        try:
            while not self.stopping and why is None:
                why = await self.segment(ch)
        except asyncio.CancelledError:
            raise
        except Exception as error:  # noqa: BLE001 - one broken chain never ends exploring
            why = f"error {type(error).__name__}"
            loop.feed.emit("explore_scan_failed", name=ch["name"], error=repr(error)[:300])
        finally:
            self.live.pop(ch["name"], None)
            if why is not None:  # a cancelled chain (stop, restart) stays in the state for the next leg
                self.done_chains += 1
                if ch["level_spent"] > 0 and why not in ("won",):
                    record_tokens(loop, ch["code"], ch["level"], ch["level_spent"], False)
                full = self.source.get(ch["code"]) in set(self.cfg["full_sources"])
                loop.feed.emit("explore_play", name=ch["name"], game=ch["code"], version=(ch["versions"] or [None])[0],
                               versions=sorted(set(ch["versions"])), levels=ch["levels"], tokens=ch["spent"],
                               outcome=why, budget=int(self.cfg["budget"]), full=full, segments=ch["segs"],
                               cleared=ch["cleared"], seconds=round(time.time() - ch["started"], 1),
                               levels_per_300k=round(ch["levels"] * 300000 / max(1, ch["spent"]), 3))

    async def segment(self, ch):
        """One segment of the chain; None to go on, else why the chain ends."""
        loop, cfg = self.loop, self.cfg
        code, level = ch["code"], int(ch["level"])
        full = self.source.get(code) in set(cfg["full_sources"])
        cap = min(int(cfg["seg_tokens"]), int(cfg["budget"]) - int(ch["spent"]))
        if not full:
            cap = min(cap, int(float(cfg["stuck"]) * self.level_cap(code, level)) - int(ch["level_spent"]))
        if cap < int(cfg["min_seg"]):
            return "budget" if full or int(cfg["budget"]) - int(ch["spent"]) < int(cfg["min_seg"]) else "stuck"
        spec = loop.v7e_level_set(code, level, fresh=ch["start"] is None)
        if spec is None:
            return "no_level_set"
        spec = {**spec, "cap": int(cap)}
        book = loop.v7["book"]
        version = book.served
        name = f"{ch['name']}-s{int(ch['segs']):03d}"
        ch["segs"] = int(ch["segs"]) + 1
        ch["versions"].append(int(version))
        starts = [] if ch["start"] is None else [(str(ch["start"][0]), int(ch["start"][1]))]
        book.acquire(version, 1)
        try:
            folder, rows = await loop.v7e_explore_play(name, spec, starts, version)
        finally:
            book.release(version, 1)
            await loop.v7_catch_up()
        row = rows[0] if rows else None
        if row is None:
            return "no_usable_row"
        if row.get("restore") and not row["restore"].get("ok"):
            return "restore_not_exact"
        tokens = int(row.get("completion_tokens") or 0)
        ch["spent"] = int(ch["spent"]) + tokens
        ch["level_spent"] = int(ch["level_spent"]) + tokens
        self.seg_tokens += tokens
        attempt = Path(folder) / "attempts" / row["label"]
        root = loop.v7e_root() / "harvest"
        loop.feed.emit("explore_seg", name=name, game=code, level=level, version=version, outcome=row.get("outcome"),
                       tokens=tokens, cap=cap)
        if row.get("outcome") in ("cleared", "won"):
            record_tokens(loop, code, level, int(ch["level_spent"]), True)
            ch["levels"] = int(ch["levels"]) + 1
            ch["cleared"].append([level, int(version), int(ch["level_spent"])])
            total = int(row.get("total_levels") or 0)
            if row.get("outcome") == "won" or (total and level >= total):
                ch["level_spent"] = 0
                return "won"
            nxt = level + 1
            target = loop.v7e_level_set(code, nxt)
            sid = f"x{int(ch['n']):04d}-{code}-L{nxt}"
            got, why = await asyncio.to_thread(cut_clear, attempt, root / sid, row, nxt,
                                               self.checks if cfg["board_check"] else None, spec["game"])
            if got is None:
                loop.feed.emit("explore_drop", name=name, level=nxt, why=why)
                ch["level_spent"] = 0
                return f"cut_{why}"
            dest, request, prompt = got
            ch.update(start=[str(dest), int(request)], level=nxt, level_spent=0)
            ceiling = cfg["max_prompt"] or (target or {}).get("max_prompt")
            if ceiling and prompt > int(ceiling):  # too long a context to train on; the chain plays on from it
                loop.feed.emit("explore_drop", name=name, level=nxt, why="prompt_over_ceiling")
            elif target is not None and target.get("role") == "train" and not target.get("fresh"):
                self.add([{"id": sid, "round": int(version), "set": None, "target": target["name"],
                           "game": target["game"], "level": nxt, "attempt": str(dest), "request": int(request),
                           "prompt_tokens": prompt, "played_by": int(version), "parent": str(attempt),
                           "parent_level": level, "clear_tokens": int(ch["cleared"][-1][2]), "source": "explore",
                           "family": target.get("family") or target["game"],
                           "start": {"attempt": str(dest), "request": int(request)}}])
                if self.first_deep is None:
                    self.first_deep = round(time.time() - self.t0, 1)
                    loop.feed.emit("explore_first_deep", seconds=self.first_deep, id=sid, version=version)
            return None
        if row.get("outcome") == "budget":
            after = int(ch["start"][1]) if ch["start"] is not None else -1
            got, why = await asyncio.to_thread(cut_mid, attempt, root / f"x{int(ch['n']):04d}-{code}-L{level}-m{int(ch['segs']):03d}", after)
            if got is None:
                return f"mid_{why}"
            ch["start"] = [str(got[0]), int(got[1])]
            return None
        return str(row.get("outcome") or "unknown")


class EnvsV7Mixin:
    """Hooks the v7 loop (arena.rl_async) calls when ``v7e_on``."""

    def v7e_on(self):
        mg = self.cfg.get("mg") or {}
        return bool(mg.get("async") and mg.get("curriculum"))

    def v7e_root(self):
        out = os.environ.get("ARC_VALIDATION_OUTPUT")
        return Path(out) if out else Path(self.output)

    # ---- sets ----
    def v7e_level_set(self, code, level, fresh=None):
        """The training set of ``code`` at ``level`` (a test set never), or None. ``fresh`` (explorer segments): a
        level-1 set for a fresh segment, else a restored-start spec of the same game at that level (the fresh L1 set
        when the chain restores into level 1)."""
        sets = self.cfg["mg_sets"]
        found = self.__dict__.setdefault("v7e_by_level", {})
        if not found:
            for n, s in sets.items():
                if s.get("role") == "train":
                    found[(code_of(s["game"]), int(s.get("level") or 1))] = n
        name = found.get((code_of(code), int(level)))
        s = sets.get(name) if name else None
        if fresh is None:
            return s
        if s is None:  # a level without a training set (no action reference): the game's spec at that level
            any_ = next((sets[n] for (c, _), n in found.items() if c == code_of(code)), None)
            if any_ is None:
                return None
            s = {**any_, "name": f"{code_of(code)}-L{int(level)}", "level": int(level)}
            s.pop("fresh", None)
        if fresh:
            return {**s, "fresh": True} if int(level) == 1 else None
        s = dict(s)
        s.pop("fresh", None)
        return s

    def v7e_active(self):
        return self.v7e["active"]

    # ---- setup / state ----
    async def v7e_setup(self, state):
        """After the book exists (v7_setup): the pool, caps, rotation and explorers, restored from ``state`` (the
        loop state's ``envs``) when resuming."""
        mg, sets = self.cfg["mg"], self.cfg["mg_sets"]
        trained = {code_of(s["game"]) for s in sets.values() if s.get("role") == "train"}
        held = {code_of(s["game"]) for s in sets.values() if s.get("role") == "test"}
        if trained & held:  # the 15/10 rule: a held-out game never trains
            raise ValueError(f"v7 envs: games both trained and held out: {sorted(trained & held)}")
        reserve = []
        for s in sets.values():
            if s.get("role") == "train" and s.get("reserve") and code_of(s["game"]) not in reserve:
                reserve.append(code_of(s["game"]))
        active = sorted(trained - set(reserve))
        self.v7e = {"active": active, "reserve": reserve, "rotated": [], "explored": {}, "added": {}, "expired": 0,
                    "groups_by_source": {}, "tokens_train": 0, "chain": {}, "gmeta": {}, "held_starts": {},
                    "src_count": {}, "chain_log": []}
        self.__dict__.setdefault("mg_harvest", [])
        self.__dict__.setdefault("mg_harvest_uses", {})
        info = {}
        if state:
            got = harvest_mod.restore_state(self, state, getattr(self, "v7_resume_root", None))
            info["restored"] = got
            self.mg_level_obs = state.get("level_obs") or {}
            for n, c in (state.get("set_caps") or {}).items():
                if c and n in sets:
                    sets[n]["cap"] = int(c)
            for k in ("active", "reserve", "rotated", "explored", "groups_by_source", "chain", "gmeta", "held_starts",
                      "wread_base"):
                if state.get(k) is not None:
                    self.v7e[k] = state[k]
            self.v7e["held_starts"] = {n: [self.v7e_found_start(x) for x in v]
                                       for n, v in (self.v7e.get("held_starts") or {}).items()}
        self.v7e_exclude_inactive()
        self.mg_explorers = None
        if mg.get("explore"):
            ex = dict((state or {}).get("explore") or {})
            ex["chains"] = [self.v7e_found_chain(c) for c in ex.get("chains") or []]
            self.mg_explorers = Chains(self, ex).start()
        self.feed.emit("v7e_setup", resume=bool(state), active=len(self.v7e["active"]), reserve=len(self.v7e["reserve"]),
                       pool=len(self.mg_harvest), chains=len(((state or {}).get("explore") or {}).get("chains") or []), **info)

    def v7e_found_start(self, start):
        """A recorded start [folder, request] from an earlier leg (whole-game reads cut them under wread/): the folder
        where this leg can read it."""
        p, root = Path(start[0]), getattr(self, "v7_resume_root", None)
        cands = [p] + ([Path(root) / "wread" / p.name, Path(root) / "harvest" / p.name] if root else [])
        if str(p).startswith("/runs/"):
            cands.append(Path("/tmp") / p.relative_to("/runs"))
        return [str(next((c for c in cands if (c / "rollout.jsonl").is_file()), p)), int(start[1])]

    def v7e_found_chain(self, ch):
        if ch.get("start"):
            p = Path(ch["start"][0])
            root = getattr(self, "v7_resume_root", None)
            cands = [p] + ([Path(root) / "harvest" / p.name] if root else [])
            if str(p).startswith("/runs/"):
                cands.append(Path("/tmp") / p.relative_to("/runs"))
            ch = {**ch, "start": [str(next((c for c in cands if (c / "rollout.jsonl").is_file()), p)), ch["start"][1]]}
        return ch

    def v7e_state(self):
        ex = self.mg_explorers
        h = harvest_mod.state(self)
        return {**h, "level_obs": getattr(self, "mg_level_obs", {}),
                "set_caps": {n: v.get("cap") for n, v in self.cfg["mg_sets"].items() if v.get("role") == "train"},
                "explore": ex.state() if ex is not None else None,
                **{k: self.v7e[k] for k in ("active", "reserve", "rotated", "explored", "groups_by_source", "chain", "gmeta",
                                            "held_starts", "wread_base") if k in self.v7e}}

    async def v7e_stop(self):
        ex = getattr(self, "mg_explorers", None)
        if ex is not None:
            try:
                await asyncio.wait_for(ex.stop(), 300)
            except Exception as error:  # noqa: BLE001 - the run's end goes on
                self.feed.emit("alert", why="explorer stop failed", error=repr(error)[:300])

    # ---- pool ----
    def v7e_fresh(self):
        """{set name: pool rows drawable now} (under max_uses, at most ``stale`` versions behind the trainer)."""
        h = harvest_mod.settings(self.cfg["mg"]) or harvest_mod.KEYS
        book, uses = self.v7["book"], self.mg_harvest_uses
        out = {}
        for r in self.mg_harvest:
            if uses.get(r["id"], 0) >= int(h["max_uses"]):
                continue
            lag = book.lag(int(r.get("played_by") or 0))
            if lag is None or (h.get("stale") is not None and lag > int(h["stale"])):
                continue
            out.setdefault(r["target"], []).append(r)
        return out

    def v7e_prune(self):
        """Spent rows and rows too old ever to be drawn again (beyond stale + s_max, a rollback's reach) leave the
        in-memory pool (harvest.jsonl keeps every row)."""
        h = harvest_mod.settings(self.cfg["mg"]) or harvest_mod.KEYS
        book, uses = self.v7["book"], self.mg_harvest_uses
        floor = book.current - int(h.get("stale") or 10 ** 6) - int(self.v7_cfg_now["s_max"])
        keep, gone = [], 0
        for r in self.mg_harvest:
            if uses.get(r["id"], 0) >= int(h["max_uses"]) or int(r.get("played_by") or 0) < floor:
                gone += 1
                continue
            keep.append(r)
        self.mg_harvest = keep
        ids = {r["id"] for r in keep}
        self.mg_harvest_uses = {k: v for k, v in self.mg_harvest_uses.items() if k in ids}
        self.v7e["expired"] += gone
        return gone

    def v7e_exclude_inactive(self):
        return {k for k, n in self.mg_unit_set.items() if code_of(self.cfg["mg_sets"][n]["game"]) not in set(self.v7e["active"])}

    def v7e_count(self, src, meta):
        self.v7e["pick_meta"] = meta
        c = self.v7e.setdefault("src_count", {})
        c[src] = c.get(src, 0) + 1

    def v7e_chain_pick(self, sets, fresh, exclude):
        """(set name, start, cap) of a chain group, or None (Curriculum.plan_chain; module doc, Chains)."""
        cur = self.mg_curr
        c = cur.cfg
        n = int(c.get("chain_groups") or 0)
        if n <= 0:
            return None
        step = int(self.v7["book"].current)
        ch = self.v7e.setdefault("chain", {})
        if ch.get("step") != step:
            ch.update(step=step, made=0, games=[], sure=0)
        if int(ch["made"]) >= n:
            return None
        closed, dry = ch.setdefault("closed", {}), ch.setdefault("dry", {})
        for k, at in list(closed.items()):
            if step - int(at) >= int(c["chain_reopen"]):
                closed.pop(k)
                dry.pop(k, None)
        budget = int(c["chain_budget"])
        rows_of, by_game = {}, {}
        for key, name in self.mg_unit_set.items():
            s = sets[name]
            lvl = int(s.get("level") or 1)
            if key in exclude or key in closed or lvl < 2 or s.get("fresh"):
                continue
            code = code_of(s["game"])
            if code in ch["games"]:
                continue
            u = cur.units.get(key)
            cap = int((u.cap if u is not None else None) or s.get("cap") or self.cfg["mg"]["cap"])
            rows = [r for r in fresh.get(name) or [] if int(r.get("chain_tokens") or 0) + cap <= budget]
            if rows:
                rows_of[key] = (rows, cap)
                by_game.setdefault(code, []).append((lvl, key))
        cands = {}
        for items in by_game.values():
            deep = max(lvl for lvl, _ in items)
            for lvl, key in items:
                cands[key] = {"deepest": lvl == deep}
        p = cur.plan_chain(step, cands, seed=int(c.get("seed", 0)), sure_ok=int(ch["sure"]) < int(c["chain_sure_max"]))
        if p is None:
            return None
        rows, cap = rows_of[p["unit"]]
        r = sorted(rows, key=lambda r: (self.mg_harvest_uses.get(r["id"], 0), int(r.get("chain_tokens") or 0),
                                        -int(r.get("played_by") or 0), r["id"]))[0]
        self.mg_harvest_uses[r["id"]] = self.mg_harvest_uses.get(r["id"], 0) + 1
        name = self.mg_unit_set[p["unit"]]
        code = code_of(sets[name]["game"])
        ch["made"] = int(ch["made"]) + 1
        ch["games"].append(code)
        ch["sure"] = int(ch["sure"]) + int(p["sure"])
        self.v7e.setdefault("chain_log", []).append([code, int(p["level"]), bool(p["sure"])])
        self.v7e_count("chain", {"chain": True, "unit": p["unit"], "tokens": int(r.get("chain_tokens") or 0)})
        u = cur.units[p["unit"]]
        return name, [r["attempt"], int(r["request"])], (int(u.cap) if u.cap else cap)

    def v7e_pick(self, sets, pool):
        """(set name, start or None, cap) of the next group, or (None, None, None)."""
        cur, mg = self.mg_curr, self.cfg["mg"]
        fresh = self.v7e_fresh()
        exclude = self.v7e_exclude_inactive()
        for key, name in self.mg_unit_set.items():
            s = sets[name]
            if not s.get("fresh") and not fresh.get(name) and not pool.get(name):
                exclude.add(key)
        got = self.v7e_chain_pick(sets, fresh, exclude)
        if got is not None:
            return got
        for attempt in range(3):
            p = cur.plan_one(self.v7["book"].current, seed=int(mg["curriculum"].get("seed", 0)), exclude=exclude,
                             salt=f"+{attempt}" if attempt else "")
            if p is None:
                return None, None, None
            name = self.mg_unit_set[p["unit"]]
            s = sets[name]
            u = cur.units[p["unit"]]
            cap = int(u.cap) if u.cap else None
            if s.get("fresh"):
                self.v7e_count("fresh", {"chain": False, "unit": p["unit"], "tokens": 0})
                return name, None, cap
            rows = sorted(fresh.get(name) or [], key=lambda r: (self.mg_harvest_uses.get(r["id"], 0),
                                                                 -int(r.get("played_by") or 0), r["id"]))
            if rows:
                r = rows[0]
                self.mg_harvest_uses[r["id"]] = self.mg_harvest_uses.get(r["id"], 0) + 1
                self.v7e_count(str(r.get("source") or "harvest"), {"chain": False, "unit": p["unit"],
                                                                  "tokens": int(r.get("chain_tokens") or 0)})
                return name, [r["attempt"], int(r["request"])], cap
            bank = list(pool.get(name) or [])
            if bank:
                k = self.v7["made"]["by_set"].get(name, 0)
                self.v7e_count("bank", {"chain": False, "unit": p["unit"], "tokens": 0})
                return name, list(bank[k % len(bank)]), cap
            cur.unplan(p)
            exclude.add(p["unit"])
        return None, None, None

    # ---- per group / per step ----
    async def v7e_group_done(self, g, spec, folder, rows):
        """A finished training group: its clears' harvest cuts (next level's starts), running caps, source count."""
        mg = self.cfg["mg"]
        src = spec.get("source") or "sets"
        self.v7e["groups_by_source"][src] = self.v7e["groups_by_source"].get(src, 0) + 1
        self.v7e["tokens_train"] += sum(int(r.get("completion_tokens") or 0) for r in rows)
        out = {}
        meta = (self.v7e.get("gmeta") or {}).pop(g.gid, None) or {}
        ch = self.v7e.setdefault("chain", {})
        clears = sum(1 for r in rows if r.get("outcome") in ("cleared", "won"))
        key = meta.get("unit")
        if key is not None and clears:  # any clear of the level reopens a closed chain level
            ch.setdefault("closed", {}).pop(key, None)
            ch.setdefault("dry", {}).pop(key, None)
        elif key is not None and meta.get("chain") and rows:
            dry = ch.setdefault("dry", {})
            dry[key] = int(dry.get(key, 0)) + 1
            if dry[key] >= int(self.mg_curr.cfg["chain_stuck"]):
                ch.setdefault("closed", {})[key] = int(self.v7["book"].current)
                self.feed.emit("chain_closed", unit=key, dry=dry[key], step=int(self.v7["book"].current))
        if meta.get("chain"):
            out["chain"] = True
        if mg.get("explore") or mg.get("running_caps") is not None:
            out["caps"] = update_caps(self, {g.unit: (folder, rows, None)}, mg.get("running_caps"))
        if harvest_mod.wants(mg, self.cfg["mg_sets"], spec) and any(r.get("harvest") for r in rows):
            hs = harvest_mod.settings(mg)
            if hs["board_check"] and getattr(self, "v7e_checks", None) is None:
                try:
                    self.v7e_checks = harvest_mod.arcade(self) or False
                except Exception as error:  # noqa: BLE001 - the restore check still runs when a start plays
                    self.v7e_checks = False
                    self.feed.emit("alert", why="harvest board check unavailable", error=repr(error)[:300])
            checks = getattr(self, "v7e_checks", None) or None
            number = int(str(g.gid).lstrip("g") or 0)  # unique per group: the cut ids never collide
            # the cut (snapshots, board replays) runs off the event loop on a copy of the pool; the new rows join the
            # live pool here (explorers append and the step prunes it meanwhile)
            shim = SimpleNamespace(cfg=self.cfg, output=self.output, feed=self.feed, mg_attempt_start=self.mg_attempt_start,
                                   mg_harvest=list(self.mg_harvest), mg_harvest_uses=dict(self.mg_harvest_uses))
            base = len(shim.mg_harvest)
            got = await asyncio.to_thread(harvest_mod.collect, shim, number, {g.unit: (folder, rows, None)},
                                          int(g.version), checks)
            new = shim.mg_harvest[base:]
            for r in new:  # chain budget: the parent start's tokens plus this clear's
                r["chain_tokens"] = int(meta.get("tokens") or 0) + int(r.get("clear_tokens") or 0)
            if new:
                self.mg_harvest.extend(new)
                cur = getattr(self, "mg_curr", None)
                for r in new:
                    if cur is not None:
                        cur.add_start(r, create=False)
            if got:
                out["harvest"] = {k: got[k] for k in ("offered", "new", "dropped")}
        return out

    def v7e_rotate(self, step):
        """Games that stopped teaching leave the active set; the next reserve game of the same source comes in."""
        r = rotation_settings(self.cfg["mg"])
        if r is None or not self.v7e["reserve"] or step % max(1, int(r["every"])):
            return []
        sets, cur = self.cfg["mg_sets"], self.mg_curr
        ex = getattr(self, "mg_explorers", None)
        explored = {}
        for c in (self.v7e.get("explored") or {}):
            explored[c] = int(self.v7e["explored"][c])
        src_of = {code_of(s["game"]): s.get("source") for s in sets.values() if s.get("role") == "train"}
        fresh = self.v7e_fresh()
        units = {}
        for key, name in self.mg_unit_set.items():
            units.setdefault(code_of(sets[name]["game"]), []).append((cur.units[key], name))
        moved = []
        for code in list(self.v7e["active"]):
            if src_of.get(code) in set(r["keep"]):
                continue
            us = units.get(code) or []
            l1 = [u for u, _ in us if u.level == 1]
            seen = [u for u, _ in us if u.seen > 0]
            open_ = [u for u, n in us if u.seen == 0 and u.state == "active" and (sets[n].get("fresh") or fresh.get(n))]

            def spent(u):
                return u.state in ("retired", "parked") or (u.tries >= int(r["min_tries"]) and (
                    u.won >= float(r["retire_p"]) * u.tries or u.won <= float(r["park_p"]) * u.tries))

            done = (l1 and l1[0].seen > 0 and all(spent(u) for u in seen)
                    and explored.get(code, 0) >= int(r["min_explored"]) and not open_)
            if not done:
                continue
            nxt = next((c for c in self.v7e["reserve"] if src_of.get(c) == src_of.get(code)), None)
            if nxt is None:
                continue
            self.v7e["reserve"].remove(nxt)
            self.v7e["active"] = sorted((set(self.v7e["active"]) - {code}) | {nxt})
            row = {"step": step, "out": code, "in": nxt, "source": src_of.get(code),
                   "states": {u.key: [u.state, u.won, u.tries] for u in seen}}
            self.v7e["rotated"].append(row)
            moved.append(row)
            self.feed.emit("rotate", **row)
        if ex is not None:
            for c in moved:
                ex.rot.append(c["in"])
        return moved

    def v7e_step_row(self, step):
        """The step's data row (feed ``envs``): pool by level depth, cuts, expiry, explorers, source mix, rotation."""
        ex = getattr(self, "mg_explorers", None)
        gone = self.v7e_prune()
        fresh = self.v7e_fresh()
        depth = {}
        for name, rows in fresh.items():
            lvl = int(self.cfg["mg_sets"][name].get("level") or 1)
            depth[str(lvl)] = depth.get(str(lvl), 0) + len(rows)
        by_src = {}
        for r in self.mg_harvest:
            by_src[r.get("source") or "?"] = by_src.get(r.get("source") or "?", 0) + 1
        seg = int(ex.seg_tokens) if ex is not None else 0
        train = int(self.v7e["tokens_train"])
        row = {"step": step, "fresh_by_level": dict(sorted(depth.items(), key=lambda kv: int(kv[0]))),
               "pool": len(self.mg_harvest), "pool_by_source": by_src, "expired": gone,
               "groups_by_source": dict(self.v7e["groups_by_source"]),
               "explore": None if ex is None else {"live": len(ex.live), "limit": getattr(ex, "_limit", None),
                                                   "chains_done": ex.done_chains, "first_deep_s": ex.first_deep,
                                                   "token_share": round(seg / max(1, seg + train), 3)},
               "active": len(self.v7e["active"]), "reserve": len(self.v7e["reserve"]),
               "start_source": dict(self.v7e.get("src_count") or {}),
               "chain_groups": [f"{c}-L{lv}{'*' if sure else ''}" for c, lv, sure in self.v7e.get("chain_log") or []],
               "chain_closed": sorted((self.v7e.get("chain") or {}).get("closed") or {})}
        self.v7e["src_count"], self.v7e["chain_log"] = {}, []
        if ex is not None:
            ex.seg_tokens = 0
        self.v7e["tokens_train"] = 0
        rotated = self.v7e_rotate(step)
        if rotated:
            row["rotated"] = rotated
        if step >= 4 and ex is not None and ex.first_deep is None and not self.v7e.get("deep_alerted"):
            self.v7e["deep_alerted"] = True  # deeper starts before step 4; L1 groups go on meanwhile
            self.feed.emit("alert", why="no explorer start past level 1 by step 4 (groups fall back to level 1)", step=step)
        self.feed.emit("envs", **row)
        return row

    async def v7e_explore_play(self, name, spec, starts, version, wall=None, floor=True, count=True):
        """One explorer segment: a one-try bench on ``version``'s model, server 1 only, floor priority, the harvest
        request owed at a clear. Returns (folder, usable rows of the game). Whole-game reads: their own ``wall``,
        normal priority (``floor`` False), no explored count."""
        model = self.v7_model(version)
        path, _ = self.mg_points(spec, starts, name, harvest=True)
        users = self.__dict__.setdefault("mg_users", {})
        users[model] = users.get(model, 0) + 1
        try:
            folder, rows, _ = await self.bench(name, [spec["game"]], 1, {spec["game"]: int(spec["cap"])}, points=path,
                                               model=model, concurrency=1,
                                               wall=int(wall or self.mg_explorers.cfg["wall"]),
                                               reserve=60, tail={"floor": True} if floor else {},
                                               own={"server1_only": True, "policy_version": int(version)})
        finally:
            users[model] -= 1
            if not users[model]:
                try:
                    await self.mg_gc_names()
                except Exception as error:  # noqa: BLE001
                    self.feed.emit("alert", why="adapter unload failed", name=model, error=repr(error)[:300])
        if count and int(spec.get("level") or 1) == 1 and not starts:
            code = code_of(spec["game"])
            self.v7e["explored"][code] = int(self.v7e["explored"].get(code, 0)) + 1
        return folder, self.game_rows(rows, spec["game"])

    # ---- whole-game held-out read (mg.wread) ----
    def v7e_wread_due(self, step):
        w = wread_settings(self.cfg["mg"])
        return w is not None and int(step) > 0 and int(w["every"]) > 0 and int(step) % int(w["every"]) == 0

    def v7e_held_set(self, code, level, fresh):
        """A held-out (test) set spec of ``code`` at ``level`` for a whole-game segment: fresh at level 1 with no
        start, else a restored-start spec (the game's spec at that level when no test set has it)."""
        tests = [x for x in self.cfg["mg_sets"].values() if x.get("role") == "test" and code_of(x["game"]) == code_of(code)]
        s = next((x for x in tests if int(x.get("level") or 1) == int(level)), None)
        if s is None:
            if not tests:
                return None
            s = {**tests[0], "name": f"{code_of(code)}-L{int(level)}", "level": int(level)}
        s = dict(s)
        s.pop("fresh", None)
        if fresh:
            return {**s, "fresh": True} if int(level) == 1 else None
        return s

    def v7e_wread_extra(self, fixed):
        """{set name: [start]} for the extended per-level read: held-out levels a whole-game read reached that the
        core read lacks (first captured start, fixed), at most ``extra_sets``."""
        w = wread_settings(self.cfg["mg"])
        if w is None:
            return {}
        sets = self.cfg["mg_sets"]
        out = {}
        for name in sorted(self.v7e.get("held_starts") or {}, key=lambda n: (n.split("-L")[0], int(n.split("-L")[1]))):
            if name in fixed or name not in sets or sets[name].get("role") != "test":
                continue
            out[name] = [tuple(self.v7e["held_starts"][name][0])]
            if len(out) >= int(w["extra_sets"]):
                break
        return out

    async def v7e_wgame(self, code, version, step, w):
        """One held-out game from level 1 for ``w.budget`` tokens on ``version`` (explorer-style segments)."""
        from .rl_single import clear_rhae

        root = self.v7e_root() / "wread"
        start, level, spent, segs, tok_level, total, why = None, 1, 0, 0, 0, None, None
        cleared, tokens = {}, {}
        while why is None:
            cap = min(int(w["seg_tokens"]), int(w["budget"]) - spent)
            if cap < int(w["min_seg"]):
                why = "budget"
                break
            spec = self.v7e_held_set(code, level, fresh=start is None)
            if spec is None:
                why = "no_level_set"
                break
            spec = {**spec, "cap": int(cap)}
            name = f"w{int(step):04d}-{code}-s{segs:03d}"
            segs += 1
            folder, rows = await self.v7e_explore_play(name, spec, [] if start is None else [tuple(start)], version,
                                                       wall=int(w["wall"]), floor=False, count=False)
            row = rows[0] if rows else None
            if row is None:
                why = "no_usable_row"
                break
            if row.get("restore") and not row["restore"].get("ok"):
                why = "restore_not_exact"
                break
            t = int(row.get("completion_tokens") or 0)
            spent, tok_level = spent + t, tok_level + t
            total = int(row.get("total_levels") or 0) or total
            attempt = Path(folder) / "attempts" / row["label"]
            if row.get("outcome") in ("cleared", "won"):
                cleared[level] = round(100.0 * (clear_rhae(row) or 0.0), 3)
                tokens[level], tok_level = tok_level, 0
                if row.get("outcome") == "won" or (total and level >= total):
                    why = "won"
                    break
                nxt = level + 1
                got, cut = await asyncio.to_thread(cut_clear, attempt, root / f"w{int(step):04d}-{code}-L{nxt}", row, nxt,
                                                   None, spec["game"])
                if got is None:
                    why = f"cut_{cut}"
                    break
                start, level = [str(got[0]), int(got[1])], nxt
                held = self.v7e.setdefault("held_starts", {})
                held.setdefault(f"{code_of(code)}-L{nxt}", [start])  # the first start captured stays the set's start
                continue
            if row.get("outcome") == "budget":
                after = int(start[1]) if start is not None else -1
                got, cut = await asyncio.to_thread(cut_mid, attempt, root / f"w{int(step):04d}-{code}-L{level}-m{segs:03d}",
                                                   after)
                if got is None:
                    why = f"mid_{cut}"
                    break
                start = [str(got[0]), int(got[1])]
                continue
            why = str(row.get("outcome") or "unknown")
        total = int(total or level)
        return {"game": code, "score": round(game_score(cleared, total), 4), "levels": len(cleared), "total_levels": total,
                "tokens": spent, "segments": segs, "why": why, "cleared": {str(k): v for k, v in cleared.items()},
                "level_tokens": {str(k): v for k, v in tokens.items()}}

    async def v7e_wread(self, step, version):
        """The whole-game read of ``version`` (step 0: the base): every held-out game at once, one try each."""
        w = wread_settings(self.cfg["mg"])
        games = sorted({code_of(x["game"]) for x in self.cfg["mg_sets"].values() if x.get("role") == "test"})
        book = self.v7["book"]
        hold = not (int(version) == 0 and self.v7.get("base0"))  # the base needs no LoRA slot (a resumed base read
        if hold:  # holding version 0 for hours would keep new versions from serving: Book.take_pending)
            book.acquire(version, len(games), read=True)
        began = time.time()
        try:
            got = await asyncio.gather(*(self.v7e_wgame(c, version, step, w) for c in games), return_exceptions=True)
        finally:
            if hold:
                book.release(version, len(games), read=True)
            await self.v7_catch_up()
        res = {}
        for c, r in zip(games, got):
            res[c] = {"error": repr(r)[:300]} if isinstance(r, BaseException) else r
        ok = [r for r in res.values() if "score" in r]
        score, se = mean_se([r["score"] for r in ok])
        tok = sum(r["tokens"] for r in ok)
        per = round(sum(r["score"] for r in ok) / (tok / 1e6), 4) if tok else None
        row = {"step": int(step), "version": int(version), "score": None if score is None else round(score, 4),
               "se": None if se is None else round(se, 4), "score_per_mtok": per, "tokens": tok, "games": res,
               "errors": len(res) - len(ok), "seconds": round(time.time() - began, 1),
               "held_starts": len(self.v7e.get("held_starts") or {})}
        if int(step) == 0:
            self.v7e["wread_base"] = {"score": score, "se": se, "per": per}
            row["base"] = True
        base = self.v7e.get("wread_base") or {}
        if base.get("score") and score is not None:  # target: 2x held-out RHAE/tok vs the base
            row["ratio_score"] = ratio_ci(score, se, base["score"], base.get("se"))
            if per and base.get("per"):
                row["ratio_per_mtok"] = round(per / base["per"], 4)
        self.feed.emit("wread", **row)
        return row

