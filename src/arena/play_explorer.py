"""Explorers (envs arm): copies of the current policy play whole games from level 1 while the
loop trains; every level they reach becomes a restored start with its real context.

- ``Explorers`` runs as one background task of the multi-game loop. It keeps ``streams`` plays going over the active
  games (round-robin, at most ``per_game`` at once per game). A play is a one-try level bench of the game's
  play-through point at ``budget`` tokens on the adapter served when it starts, which it keeps for life (loaded once
  more as explore-rNN on the play server, unloaded when no play uses it; under mg.lookahead_below the versioned
  policy-rNN, registered in ``loop.mg_users``). Games whose source is not in
  ``full_sources`` stop when stuck: the point's ``level_caps`` are ``stuck`` x each level's running cap.
- Every ``poll`` seconds the live plays are scanned: a level k >= 2 the play has reached (its first request answered)
  is snapshotted (rollout rows through that request, the log so far) to ``<root>/harvest/<id>``, checked (no unsafe
  event before it, the prompt ceiling, ``restore.load`` agrees, the CPU board check) and added to the loop's start
  pool (``loop.mg_harvest``, source "explore", ``played_by`` = the play's adapter version) and the curriculum.
- A finished play emits ``explore_play`` (the progress read: levels cleared per budget per adapter version) and its
  per-level tokens feed the running caps (``update_caps``).

``mg.explore``: {"streams", "budget", "games" (codes), "per_game", "full_sources", "stuck", "poll", "max_prompt",
"board_check", "wall", "default_cap"}.
"""
from __future__ import annotations

import asyncio
import json
import os
import shutil
import statistics
import time
from collections import deque
from pathlib import Path

KEYS = {"streams": 24, "budget": 300000, "games": [], "per_game": 2, "full_sources": ["public"], "stuck": 2.0,
        "poll": 60, "max_prompt": None, "board_check": True, "wall": 14400, "default_cap": 64000,
        "kv_share": 0.15, "context": 131072, "nodata_cap": 27000,
        "seg_tokens": 65536, "min_seg": 2000,
        # "unplayed_first" = never-played active games, then ``candidates`` (reserve
        # games with short base-measured sets, envs) not yet explored, then the usual rotation
        "order": "round", "candidates": []}  # v7 chains (arena.v7_envs): a segment's tokens, the least worth playing
# nodata_cap: a level without clears stops a stuck explorer at stuck x 27K (1.25 x p80 of the env
# probe's per-level median tokens to clear on L2+, 695 levels: p80 21.5K); its training cap stays the pool's
CAP_KEYS = {"factor": 1.25, "q": 0.8, "min_clears": 3, "min": 4000, "max": 262144}


def settings(mg):
    x = mg.get("explore")
    if not x:
        return None
    unknown = set(x) - set(KEYS)
    if unknown:
        raise ValueError(f"mg.explore: unknown keys {sorted(unknown)}")
    out = {**KEYS, **x}
    if out["order"] not in ("round", "unplayed_first"):
        raise ValueError(f"mg.explore.order {out['order']!r}: round | unplayed_first")
    if not out["games"] or int(out["streams"]) < 1 or int(out["per_game"]) < 1:
        raise ValueError("mg.explore: games (non-empty), streams >= 1, per_game >= 1")
    return out


def quantile(xs, q):
    xs = sorted(xs)
    return xs[int(q * (len(xs) - 1))]


def running_cap(obs, cfg=None):
    """A level's cap from its clears: ``factor`` x the ``q`` quantile once ``min_clears`` exist, else None (the level
    keeps its prior cap). Failed tokens (``censored``) never set a cap: a stuck stop at 2x the cap would feed back
    into a doubling cap."""
    c = {**CAP_KEYS, **(cfg or {})}
    clears = obs.get("clears") or []
    if len(clears) < int(c["min_clears"]):
        return None
    return int(min(c["max"], max(c["min"], round(c["factor"] * quantile(clears, c["q"]), -3))))


def record_tokens(loop, game, level, tokens, cleared):
    obs = loop.__dict__.setdefault("mg_level_obs", {}).setdefault(f"{str(game)[:4]}-L{int(level)}",
                                                                    {"clears": [], "censored": []})
    obs["clears" if cleared else "censored"].append(int(tokens))
    for k in ("clears", "censored"):
        del obs[k][:-200]


def update_caps(loop, plays=None, cfg=None):
    """Running caps: this round's training clears join the evidence; every training set with a cap
    from the evidence takes it (times its curriculum unit's tier factor). Returns {set: cap} changed."""
    from .rl_single import clear_tokens

    sets = loop.cfg["mg_sets"]
    for name, (_, rows, _) in (plays or {}).items():
        s = sets.get(name) or {}
        for r in rows:
            t = clear_tokens(r)
            if t:
                record_tokens(loop, s.get("game", name[:4]), s.get("level", 1), t, True)
    changed = {}
    cur = getattr(loop, "mg_curr", None)
    tier = (cur.cfg.get("tier_factor", 1.5) if cur is not None else 1.5)
    for name, s in sets.items():
        if s.get("role") != "train":
            continue
        cap = running_cap(loop.__dict__.get("mg_level_obs", {}).get(f"{s['game'][:4]}-L{int(s.get('level') or 1)}", {}), cfg)
        if cap is None or cap == int(s.get("cap") or 0):
            continue
        s["cap"] = cap
        changed[name] = cap
        if cur is not None:
            key = cur.key(s["game"], int(s.get("level") or 1))
            u = cur.units.get(key)
            if u is not None:
                u.cap = int(round(cap * tier ** int(u.tier or 0)))
                ceiling = (cur.cfg or {}).get("cap_ceiling")
                if ceiling:  # the cap ceiling holds for tiers too
                    u.cap = min(int(ceiling), u.cap)
    return changed


def snapshot(attempt, dest, request):
    """A live play cut as a restorable recording: rollout rows 0..request, the log so far (complete lines)."""
    attempt, dest = Path(attempt), Path(dest)
    rows = (attempt / "rollout.jsonl").read_bytes()
    rows = rows[: rows.rfind(b"\n") + 1].splitlines(keepends=True)[: int(request) + 1]
    log = (attempt / "log.jsonl").read_bytes()
    tmp = dest.with_name(dest.name + ".tmp")
    shutil.rmtree(tmp, ignore_errors=True)
    tmp.mkdir(parents=True)
    (tmp / "rollout.jsonl").write_bytes(b"".join(rows))
    (tmp / "log.jsonl").write_bytes(log[: log.rfind(b"\n") + 1])
    shutil.rmtree(dest, ignore_errors=True)
    tmp.rename(dest)
    return dest


class Explorers:
    def __init__(self, loop):
        self.loop = loop
        self.cfg = settings(loop.cfg["mg"])
        self.rot = deque(self.cfg["games"])
        self.live = {}  # bench name -> {"game", "code", "model", "version", "task", "started", "seen"}
        done = [int(str(r["id"])[1:5]) for r in getattr(loop, "mg_harvest", None) or []
                if r.get("source") == "explore" and str(r.get("id", ""))[1:5].isdigit()]
        self.count = max([int(getattr(loop, "mg_explore_count", 0))] + done)  # a resumed leg's ids never repeat
        self.task, self.stopping = None, False
        self.checks = None
        self.own = {}  # explore-rNN adapter names this explorer loaded: {name: path}
        reg = {str(s["game"])[:4]: s for s in loop.cfg["mg_sets"].values()}
        self.source = {c: (reg.get(c) or {}).get("source") for c in self.cfg["games"]}

    # ---- scheduling ----
    def next_game(self):
        busy = {}
        for p in self.live.values():
            busy[p["code"]] = busy.get(p["code"], 0) + 1
        for _ in range(len(self.rot)):
            code = self.rot[0]
            self.rot.rotate(-1)
            if busy.get(code, 0) < int(self.cfg["per_game"]):
                return code
        return None

    def point(self, code):
        """The game's play-through point; a stuck stop per level unless the game plays its full budget."""
        p = dict(self.loop.mg_point(code))
        p.update(id=f"{code}-game", through=True, level_index=0, level=1, prefix=[], prefix_actions=0)
        p.pop("budget", None)
        if self.source.get(code) not in set(self.cfg["full_sources"]):
            sets = self.loop.cfg["mg_sets"]
            obs = self.loop.__dict__.get("mg_level_obs") or {}
            nodata = int(self.cfg["nodata_cap"])

            def cap(s):  # a level with clears (probe prior or >= min_clears in this run): its cap; else the no-data cap
                seen = len((obs.get(f"{code}-L{int(s.get('level') or 1)}") or {}).get("clears") or [])
                has = int((s.get("prior") or {}).get("clears") or 0) > 0 or seen >= CAP_KEYS["min_clears"]
                return int(s.get("cap") or nodata) if has else nodata

            caps = {int(s.get("level") or 1): cap(s) for s in sets.values()
                    if str(s["game"])[:4] == code and s.get("role") == "train"}
            n = int(p.get("total_levels") or max(caps or [1]))
            p["level_caps"] = [int(float(self.cfg["stuck"]) * caps.get(k, nodata)) for k in range(1, n + 1)]
        else:
            p.pop("level_caps", None)
        return p

    async def model(self):
        """(served name, version) a new play keeps for life: the base, the versioned policy-rNN (mg.lookahead_below),
        or else the served adapter loaded once more under its own name explore-rNN on the play server (so the loop's
        "policy" moves on under the play); an explore-rNN no live play uses is unloaded."""
        loop = self.loop
        if loop.serving_adapter is None:
            return loop.server.model, 0
        if getattr(loop, "mg_versioned", lambda: False)():
            name = loop.mg_name()
            return name, loop.mg_play_version(name)
        version = int(getattr(loop, "mg_version", 0))
        name = f"explore-r{version:02d}"
        if name not in getattr(loop.server, "names", set()):
            seconds = await loop.server.load(name, str(loop.serving_adapter))
            self.own[name] = str(loop.serving_adapter)
            loop.feed.emit("explore_adapter", name=name, version=version, load_seconds=seconds)
        return name, version

    async def release(self):
        """Unload own adapter names no live play uses (the current version's stays for the next launch)."""
        loop = self.loop
        current = f"explore-r{int(getattr(loop, 'mg_version', 0)):02d}"
        busy = {p["model"] for p in self.live.values()}
        for name in [n for n in self.own if n not in busy and n != current]:
            try:
                await loop.server.unload(name)
            except Exception as error:  # noqa: BLE001 - a failed unload only costs a LoRA slot
                loop.feed.emit("explore_scan_failed", name=name, error=repr(error)[:300])
            self.own.pop(name, None)

    async def play(self, name, code):
        loop, rec = self.loop, self.live[name]
        users = loop.__dict__.setdefault("mg_users", {})
        users[rec["model"]] = users.get(rec["model"], 0) + 1
        path = Path(loop.work) / f"{name}-points.json"
        point = self.point(code)
        path.write_text(json.dumps({"schema": 1, "points": [point]}))
        try:
            folder, rows, play = await loop.bench(name, [code], 1, {code: int(self.cfg["budget"])},
                                                  model=rec["model"], points=path, concurrency=1,
                                                  wall=int(self.cfg["wall"]), reserve=60, tail={"floor": True},
                                                  own={"server1_only": True})  # its adapter lives on the play server
            rec["folder"] = folder
            await self.collect_one(name, rec, final=True)
            row = rows[0] if rows else {}
            done = 0
            for lr in row.get("level_rows") or []:
                record_tokens(loop, code, lr["level"], lr.get("tokens") or 0, True)
                done = max(done, int(lr["level"]))
            spent = int(row.get("completion_tokens") or 0)
            left = spent - sum(int(lr.get("tokens") or 0) for lr in row.get("level_rows") or [])
            total = int(row.get("total_levels") or 0)
            if row and left > 0 and (not total or done < total):
                record_tokens(loop, code, done + 1, left, False)
            loop.feed.emit("explore_play", name=name, game=code, version=rec["version"], model=rec["model"],
                           levels=done, tokens=spent, outcome=row.get("outcome"), budget=int(self.cfg["budget"]),
                           full=self.source.get(code) in set(self.cfg["full_sources"]),
                           starts=sorted(rec["seen"]), seconds=round(time.time() - rec["started"], 1),
                           play=play)
        finally:
            users[rec["model"]] = max(0, users.get(rec["model"], 1) - 1)
            self.live.pop(name, None)
            self.reserve()

    def limit(self):
        """Live plays allowed: ``streams``, bounded so plays x ``context`` stay within ``kv_share`` of the play
        server's KV gate (RL review 22 Oct); 1 until a bench has measured the gate."""
        from . import level_bench

        pool = level_bench.SHARED_SLOTS
        n = int(self.cfg["streams"])
        if pool is None or not pool.capacity:
            return min(n, 1)
        fit = int(float(self.cfg["kv_share"]) * pool.capacity // max(1, pool._units(int(self.cfg["context"]))))
        got = max(1, min(n, fit))
        if got != getattr(self, "_limit", None):
            self._limit = got
            self.loop.feed.emit("explore_limit", streams=n, allowed=got, kv_share=float(self.cfg["kv_share"]),
                                capacity=pool.capacity, context_units=pool._units(int(self.cfg["context"])))
        return got

    def reserve(self):
        """The live plays' KV in the shared pool's kv_fill gate: each at the window's saw-tooth context
        (KV_PROJECT ctx_frac x context, no prefix shared between plays)."""
        from . import level_bench

        pool = level_bench.SHARED_SLOTS
        if pool is not None:
            pool.reserved = float(len(self.live)) * float(level_bench.KV_PROJECT["ctx_frac"]) * int(self.cfg["context"])

    async def launch(self):
        await self.release()
        self.reserve()
        while len(self.live) < self.limit() and not self.stopping:
            code = self.next_game()
            if code is None:
                return
            self.count += 1
            self.loop.mg_explore_count = self.count
            name = f"explore-{self.count:04d}-{code}"
            model, version = await self.model()
            rec = {"game": code, "code": code, "n": self.count, "gid": self.loop.mg_point(code)["game"],
                   "model": model, "version": version,
                   "started": time.time(), "seen": set(), "folder": Path(self.loop.output) / "rl" / name}
            self.live[name] = rec
            self.reserve()
            rec["task"] = asyncio.ensure_future(self.play(name, code))

    # ---- starts ----
    def collect_one_sync(self, name, rec, final=False):
        """New level-start rows of one live play (``final``: rescan the finished play in full)."""
        from .restore import level_starts, load, unsafe_from

        loop = self.loop
        out = []
        attempts = Path(rec["folder"]) / "attempts"
        if not attempts.is_dir():
            return out
        sets = loop.cfg["mg_sets"]
        names = {int(s.get("level") or 1): n for n, s in sets.items()
                 if str(s["game"])[:4] == rec["code"] and s.get("role") == "train" and not s.get("fresh")}
        root = Path(os.environ.get("ARC_VALIDATION_OUTPUT") or loop.output) / "harvest"
        for attempt in sorted(attempts.iterdir()):
            st = rec.setdefault("scan", {}).setdefault(attempt.name, {"off": 0, "marks": 0, "pending": False})
            try:  # read only the log's new bytes; parse the rollout only when a new level mark (or a pending one) is due
                with open(attempt / "log.jsonl", "rb") as file:
                    file.seek(st["off"])
                    chunk = file.read()
                cut = chunk.rfind(b"\n") + 1
                st["off"] += cut
                st["marks"] += chunk[:cut].count(b'{"kind":"level_bench_level"')
            except OSError:
                continue
            if not st["pending"] and st["marks"] <= len([x for x in rec["seen"] if x >= 2]) and not final:
                continue
            try:
                starts = level_starts(attempt)
            except (OSError, ValueError, KeyError, StopIteration):
                st["pending"] = True
                continue
            st["pending"] = len(starts) - 1 < st["marks"]  # a level reached whose first request is not answered yet
            for request, done, step in starts:
                level = int(done) + 1
                if level < 2 or level in rec["seen"]:
                    continue
                if level not in names:  # no training set at this depth: handled, never a start
                    rec["seen"].add(level)
                    continue
                sid = f"x{rec['n']:04d}-{rec['code']}-L{level}"
                dest = root / sid
                try:
                    snapshot(attempt, dest, request)
                    info = load(dest, int(request))
                    first, _ = unsafe_from(dest)
                    spec = sets[names[level]]
                    ceiling = self.cfg["max_prompt"] or spec.get("max_prompt")
                    prompt = int(json.loads((dest / "rollout.jsonl").read_text().splitlines()[int(request)])
                                 .get("usage", {}).get("prompt_tokens") or 0)
                    why = None
                    if info["level"] != level:
                        why = "level_mismatch"
                    elif first is not None and first <= int(step):
                        why = "unsafe_event"
                    elif ceiling and prompt > int(ceiling):
                        why = "prompt_over_ceiling"
                    elif self.cfg["board_check"] and self.checks is not None:
                        from .harvest import board_check

                        ok, detail = board_check(*self.checks, rec["gid"], dest, int(request))
                        why = None if ok else "board_check"
                except (OSError, ValueError, KeyError, IndexError) as error:
                    why = f"snapshot_{type(error).__name__}"
                rec["seen"].add(level)
                if why:
                    shutil.rmtree(dest, ignore_errors=True)
                    loop.feed.emit("explore_drop", name=name, level=level, why=why)
                    continue
                out.append({"id": sid, "round": int(getattr(loop, "mg_round_now", 0)), "set": None,
                            "target": names[level], "game": spec["game"], "level": level, "attempt": str(dest),
                            "request": int(request), "prompt_tokens": prompt, "played_by": int(rec["version"]),
                            "parent": str(attempt), "parent_level": level - 1, "clear_tokens": None,
                            "source": "explore", "family": spec.get("family") or spec["game"],
                            "start": {"attempt": str(dest), "request": int(request)}})
        return out

    async def collect_one(self, name, rec, final=False):
        new = await asyncio.to_thread(self.collect_one_sync, name, rec, final)
        self.add(new)
        return new

    def add(self, new):
        if not new:
            return
        loop = self.loop
        loop.__dict__.setdefault("mg_harvest", []).extend(new)
        cur = getattr(loop, "mg_curr", None)
        for r in new:
            if cur is not None:
                cur.add_start(r, create=False)
        out = Path(os.environ.get("ARC_VALIDATION_OUTPUT") or loop.output)
        for folder in {Path(loop.output) / "rl", out}:
            try:
                folder.mkdir(parents=True, exist_ok=True)
                with open(folder / "harvest.jsonl", "a") as file:
                    for r in new:
                        file.write(json.dumps(r) + "\n")
            except OSError:
                pass
        loop.feed.emit("explore_starts", new=len(new), by_set={r["target"]: 1 for r in new},
                       versions=sorted({r["played_by"] for r in new}))

    # ---- lifecycle ----
    async def run(self):
        from .harvest import arcade

        if self.cfg["board_check"]:
            try:
                self.checks = arcade(self.loop)
            except Exception as error:  # noqa: BLE001 - no board check: starts are still restore-checked when played
                self.loop.feed.emit("alert", why="explorer board check unavailable", error=repr(error)[:300])
        self.loop.feed.emit("explore_start", streams=int(self.cfg["streams"]), games=len(self.cfg["games"]),
                            budget=int(self.cfg["budget"]))
        while not self.stopping:
            try:
                await self.launch()
            except Exception as error:  # noqa: BLE001 - e.g. an adapter load failed: retried next poll
                self.loop.feed.emit("explore_scan_failed", name="launch", error=repr(error)[:300])
            await asyncio.sleep(float(self.cfg["poll"]))
            for name, rec in list(self.live.items()):
                try:
                    await self.collect_one(name, rec)
                except Exception as error:  # noqa: BLE001 - one bad scan never ends exploring
                    self.loop.feed.emit("explore_scan_failed", name=name, error=repr(error)[:300])

    def start(self):
        self.task = asyncio.ensure_future(self.run())
        return self

    async def stop(self):
        """Verified stop: no new plays, live plays cancelled (their benches record censored tries)."""
        self.stopping = True
        tasks = [p["task"] for p in self.live.values() if p.get("task")] + ([self.task] if self.task else [])
        for t in tasks:
            t.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
        self.loop.feed.emit("explore_stop", plays=self.count, live_cancelled=len(tasks) - (1 if self.task else 0))


def progress(events, version_key="version"):
    """The progress read from explore_play rows: per adapter version, full-budget plays' mean levels cleared."""
    out = {}
    for e in events:
        if e.get("kind") != "explore_play" or not e.get("full"):
            continue
        v = out.setdefault(str(e.get(version_key)), [])
        v.append(int(e.get("levels") or 0))
    return {k: {"plays": len(v), "levels_mean": round(statistics.mean(v), 3)} for k, v in sorted(out.items())}
