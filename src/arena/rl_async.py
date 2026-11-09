"""v7 asynchronous RL loop (docs/rl-plan.md, v7): no round barrier. Groups of ``group_size`` tries start continuously
on server 1, each pinned for life to the adapter version served when it was made; finished groups wait in a buffer;
the trainer accumulates each group as it lands (one forward/backward at the current state, per-group balance) and
takes one optimizer step per ``batch_groups`` groups, then publishes the next version. Held-out reads run every
``read_every`` steps as a background stream pinned to that step's version. Every step writes a checkpoint and the
loop state (``v7_state.json``); a resume restores both. Mode B (card 2 play blocks): ``mg.v7.card2``.

Turned on by ``mg.async`` (rl_single.mg_loop hands over after its setup). Core logic: arena.async_rl."""
from __future__ import annotations

import asyncio
import json
import os
import shutil
import time
from pathlib import Path

import httpx

from .async_rl import Book, Buffer, Card2, Group, Reads, Scheduler, lag_hist

CURRICULUM_LIVE = ("retire_p", "token_p", "token_share", "family_cap", "chain_bonus", "probe_share", "master_cap",
                   "master_rhae", "token_cost", "deep_share", "deep_level")
V7_DEFAULTS = {"group_size": 10, "batch_groups": 4, "s_max": 8, "max_ready": 8, "ramp_s": 90.0, "in_flight": 110,
               "in_flight_mode": "planned", "knob_poll_s": 30.0,  # "unfinished": a running group counts only its tries not finished (v6e)
               "read_every": 8, "read_tries": 32, "k3_windows": 16, "warmup": 2, "max_steps": 16, "max_loras": 6,
               "serve_every": 1, "poll_s": 5.0, "card2": None, "ahead": False,
               # held-out read (envs, hero run): tries per recorded start (0: read_tries split over the set's starts),
               # a fresh set's tries, the read's share of active_tries (its concurrent tries, inside the pool), and a
               # version-0 read of the base at a fresh start
               "read_per_start": 0, "read_fresh_tries": 0, "read_share": 0.0, "read_base": False,
               # final reads' crash ceiling (hero-v5 A2's slow last reads): 3x the slowest read this
               # run measured, else final_read_s (ovf3t2's 32-try held-out round 1,488 s x 3), never under the floor
               "final_read_s": 4500.0, "final_read_floor_s": 600.0,
               "read_slots": 1,  # LoRA slots for reads beyond max_loras (a read never stops a new version serving)
               # training groups take free slots oldest first (v6g's resume replayed 14 groups in
               # random slot order, each finishing at 33-80 min); held-out reads stay above every group
               "oldest_first": True, "read_kv_share": False}


class StepFailed(RuntimeError):
    pass


class AsyncMixin:
    """Loop mixin: ``mg_async_run`` replaces the round loop when ``mg.async`` is on (Mode B: ModeBMixin)."""

    def mg_versioned(self):
        return bool((self.cfg.get("mg") or {}).get("async")) or super().mg_versioned()

    def v7_cfg(self):
        mg = self.cfg["mg"]
        out = {**V7_DEFAULTS, **(mg.get("v7") or {})}
        if os.environ.get("ARC_RL1_MAX_STEPS"):  # launch-time ceiling (the step count, never wall time)
            out["max_steps"] = int(os.environ["ARC_RL1_MAX_STEPS"])
        elif os.environ.get("ARC_RL1_MAX_ROUNDS"):  # the launcher's ceiling (CPU legs; v6f legs D5/R4 ran to 45 min)
            out["max_steps"] = int(os.environ["ARC_RL1_MAX_ROUNDS"])
        unknown = set(out) - set(V7_DEFAULTS)
        if unknown:
            raise ValueError(f"mg.v7: unknown keys {sorted(unknown)}")
        stub = os.environ.get("ARC_PREFLIGHT_STUB") or ""
        if stub.startswith("{") and json.loads(stub).get("shrink"):  # CPU preflight: every path at a fraction of size
            out.update(group_size=2, batch_groups=2, in_flight=8, max_ready=4, ramp_s=0.0, read_every=2, read_tries=2,
                       k3_windows=2, poll_s=1.0,
                       # v6j pfR2/R3: open final reads outwaited the 900 s stall killer (exit 3, no end-of-run verdict)
                       final_read_s=300.0, final_read_floor_s=0.0)
            if out.get("read_per_start"):
                out.update(read_per_start=1, read_fresh_tries=min(2, int(out.get("read_fresh_tries") or 2)))
            if out.get("card2"):
                out["card2"] = {**out["card2"], "min_block_s": 30.0, "max_block_s": 60.0, "ratio_on": 0.0,
                                "horizon_s": 600.0, "switch_s": 0.0, "idle_min_s": 10.0}
        return out

    def v7_slots(self, A):
        """Server 1's attempt slots. A CPU preflight shrink with Mode B caps them at ``in_flight``: server 2 takes only
        tries server 1 cannot, and 128 slots for 8-12 tries never fill (xv7c pfA: 4 blocks served 0)."""
        slots = int(self.cfg["mg"]["active_tries"])
        stub = os.environ.get("ARC_PREFLIGHT_STUB") or ""
        if stub.startswith("{") and json.loads(stub).get("shrink") and (A.get("card2") or {}).get("on"):
            slots = min(slots, int(A["in_flight"]))
        return slots

    # ---- state -------------------------------------------------------------------------------------------------

    def v7_state_dir(self, step):
        return Path(os.environ["ARC_VALIDATION_OUTPUT"]) / "state" / f"s{int(step):04d}"

    def v7_write_state(self, step, ckpt):
        return self.v7_put_state(self.v7_state_dict(step, ckpt))

    def v7_state_dict(self, step, ckpt):
        """The loop state at this step (taken now: under mg.async_checkpoint it is written once the folder lands)."""
        v = self.v7
        running = [g.__dict__ | {"status": "running"} for g in v["running"].values()]
        state = {"step": int(step), "checkpoint": str(ckpt), "book": v["book"].state_dict(),
                 "buffer": v["buffer"].state_dict(), "reads": v["reads"].state_dict(), "card2": v["card2"].state_dict(),
                 "running": running, "made": v["made"], "since_resume": v["since_resume"],
                 "base_read_done": bool(v.get("base_read_done")),
                 "adapters": {str(k): p for k, p in v["adapters"].items()},
                 "curriculum": (self.mg_curr.state_dict() if getattr(self, "mg_curr", None) is not None else None),
                 "envs": self.v7e_state() if getattr(self, "v7e", None) is not None else None,
                 "written": round(time.time(), 1)}
        return json.loads(json.dumps(state))  # a copy: later steps mutate the live structures

    def v7_put_state(self, state):
        ckpt = state["checkpoint"]
        out = Path(os.environ["ARC_VALIDATION_OUTPUT"])
        tmp = out / "v7_state.json.tmp"
        tmp.write_text(json.dumps(state))
        os.replace(tmp, out / "v7_state.json")  # a crash mid-write leaves the previous state whole
        if Path(ckpt).is_dir():  # the step's own snapshot beside its checkpoint: rl_rollback's exact resume point
            (Path(ckpt) / "v7_state.json").write_text(json.dumps(state))
        return state

    def v7_resume_state(self):
        roots = [x for x in os.environ.get("ARC_RL1_RESUME", "").split(",") if x]
        mg = self.cfg["mg"]
        for root in ([mg["resume"]] if mg.get("resume") else []) + roots[::-1]:
            p = Path(root) / "v7_state.json"
            if p.is_file():
                self.v7_resume_root = Path(root)
                return json.loads(p.read_text())
        return None

    def v7_found(self, path):
        """A path the earlier leg recorded, as this container sees it: its trial outputs (/tmp/arena/...) under the
        resumed root's arena/, and a /runs/<rel> dir unpacked from a preflight pack under /tmp/<rel>."""
        p = Path(path)
        if p.exists():
            return str(p)
        root = getattr(self, "v7_resume_root", None)
        tries = []
        if root is not None and str(p).startswith("/tmp/arena/"):
            tries.append(root / "arena" / p.relative_to("/tmp/arena"))
        if str(p).startswith("/runs/"):
            tries.append(Path("/tmp") / p.relative_to("/runs"))
        if root is not None and "state" in p.parts:  # the leg's own state/sNNNN beside its v7_state.json
            tries.append(root / Path(*p.parts[p.parts.index("state"):]))
        return str(next((t for t in tries if t.exists()), p))

    # ---- groups ------------------------------------------------------------------------------------------------

    def v7_model(self, version):
        return self.server.model if int(version) == 0 and self.v7["base0"] else self.mg_name(version)

    def v7_pick(self, sets, pool):
        """The next group's (set name, start): the curriculum's next unit (mg.curriculum), else the training sets in
        turn, each from its pool via mg_pick (rotation by the set's own group count)."""
        made = self.v7["made"]
        cur = getattr(self, "mg_curr", None)
        if cur is not None and getattr(self, "v7e", None) is not None:  # envs (hero run): arena.v7_envs
            name, start, cap = self.v7e_pick(sets, pool)
            self.v7["pick_cap"] = cap
            return name, start
        if cur is not None:
            picks = cur.plan(made["n"], seed=int(self.cfg["mg"]["curriculum"].get("seed", 0)), groups=1)
            for p in picks:
                name = self.mg_unit_set.get(p["unit"])
                if name is None:
                    continue
                got = self.mg_pick(made["by_set"].get(name, 0) + 1, name, {**sets[name], "train_starts": 1}, pool)
                if got and got[0]:
                    return name, list(got[0][0])
            return None, None
        names = [n for n, s in sets.items() if s["role"] == "train" and not s.get("branch_of")]
        for k in range(len(names)):
            name = names[(made["n"] + k) % len(names)]
            s = sets[name]
            if s.get("fresh"):
                return name, None
            got = self.mg_pick(made["by_set"].get(name, 0) + 1, name, {**s, "train_starts": 1}, pool)
            if got and got[0]:
                return name, list(got[0][0])
        return None, None

    def v7_make_group(self, sets, pool):
        v, A = self.v7, self.v7_cfg_now
        name, start = self.v7_pick(sets, pool)
        if name is None:
            return None
        v["made"]["n"] += 1
        v["made"]["by_set"][name] = v["made"]["by_set"].get(name, 0) + 1
        # the effective cap is pinned on the group (v6i 24 Oct: running caps moved 27K->26K between legs, so replays
        # took the new set cap, the plan digests changed and 9 of 11 groups lost every kept and mid-try snapshot)
        cap = int(v.pop("pick_cap", None) or sets[name].get("cap") or self.cfg["mg"]["cap"])
        g = Group(gid=f"g{v['made']['n']:05d}", unit=name, start=start, version=v["book"].served,
                  size=int(A["group_size"]), created=time.time(), cap=cap)
        v["running"][g.gid] = g
        if getattr(self, "v7e", None) is not None:  # envs: the pick's chain meta rides with the group (v7_envs)
            meta = self.v7e.pop("pick_meta", None)
            if meta:
                self.v7e.setdefault("gmeta", {})[g.gid] = meta
        v["tasks"][g.gid] = asyncio.ensure_future(self.v7_play_group(g, self.v7_spec(g)))
        v["scheduler"].made(time.time())
        self.feed.emit("group_start", gid=g.gid, set=name, start=start, version=g.version,
                       **({"cap": cap} if cap is not None else {}))
        return g

    def v7_spec(self, g):
        """A group's set spec at the group's own cap (the curriculum's running cap or tier, when it differs)."""
        spec = self.cfg["mg_sets"][g.unit]
        return {**spec, "cap": int(g.cap)} if getattr(g, "cap", None) else spec

    def v7_reload_marker(self):
        """mg.reload_check_once: a passed reload check's record per trainer key (ARC_RL1_TRAINER_KEY: trainer code and
        command) on the volume; true = $ARC_RELOAD_CHECK_ROOT or /runs/reload-checks. None: check every leg."""
        root, key = self.cfg["mg"].get("reload_check_once"), os.environ.get("ARC_RL1_TRAINER_KEY")
        if not root or not key:
            return None
        if root is True:
            root = os.environ.get("ARC_RELOAD_CHECK_ROOT") or "/runs/reload-checks"
        import hashlib
        return Path(str(root)) / f"{hashlib.sha256(key.encode()).hexdigest()[:32]}.json"

    def v7_reload_known(self, step):
        """A restart on a trainer key whose reload check already passed skips it (v6h: checkpoint join 83 s + check
        101 s held the first accumulate after every resume)."""
        f = self.v7_reload_marker()
        if f is None or not f.is_file():
            return False
        try:
            rec = json.loads(f.read_text())
        except ValueError:
            return False
        self.feed.emit("reload_check_skip", step=step, marker=str(f), passed=rec)
        return True

    def v7_reload_mark(self, step):
        f = self.v7_reload_marker()
        if f is None:
            return
        try:
            f.parent.mkdir(parents=True, exist_ok=True)
            f.write_text(json.dumps({"step": step, "at": time.time(), "run": os.environ.get("ARC_VALIDATION_OUTPUT")}))
        except OSError as exc:  # a speed-up only: the next leg checks again
            self.feed.emit("alert", why="reload check marker not written", error=repr(exc)[:200])

    def v7_keep(self):
        """Training groups' finished tries kept as they end (level_bench keep_try, v6's partial/ mechanism) under
        partial/v7/<gid>-<set>; a resumed leg's replay of that group takes them instead of playing them again.
        mg.try_keep false turns it off."""
        out = os.environ.get("ARC_VALIDATION_OUTPUT")
        if not out or self.cfg["mg"].get("try_keep", True) is False:
            return None
        roots = [x for x in os.environ.get("ARC_RL1_RESUME", "").split(",") if x]
        return {"try_keep": {"keep": str(Path(out) / "partial" / "v7"),
                             "reuse": [str(Path(r) / "partial" / "v7") for r in roots],
                             # resume mid-try (b): unfinished tries snapshotted every try_snap_s (0: off)
                             "snap_s": float(self.cfg["mg"].get("try_snap_s", 300))}}

    def v7_kept_cap(self, g):
        """The token budget a group's kept tries (earlier legs' partial/v7) played at, or None."""
        for root in reversed((self.v7_keep() or {}).get("try_keep", {}).get("reuse") or []):
            try:
                with open(Path(root) / f"{g.gid}-{g.unit}" / "tries.jsonl") as f:
                    for line in f:
                        budget = json.loads(line).get("token_budget")
                        if budget:
                            return int(budget)
            except (OSError, ValueError):
                continue
        return None

    def v7_own(self, g):
        own = dict(self.v7_keep() or {})
        if self.v7_cfg_now.get("oldest_first"):  # level_bench: slot priority by group number (gids survive a resume)
            own["admit_order"] = int(g.gid.lstrip("g"))
        return own or None

    async def v7_play_group(self, g, spec):
        mg, v = self.cfg["mg"], self.v7
        from . import harvest as harvest_mod

        v["book"].acquire(g.version, g.size)
        began = time.time()
        try:
            folder, rows, _ = await self.mg_play(f"{g.gid}-{g.unit}", spec, [tuple(g.start)] if g.start else [], g.size,
                                                 model=self.v7_model(g.version), seed=None,
                                                 sampling=mg.get("train_sampling"),
                                                 harvest=harvest_mod.wants(mg, self.cfg["mg_sets"], spec),
                                                 own=self.v7_own(g))
        finally:
            v["book"].release(g.version, g.size)
            await self.v7_catch_up()
        played = [int(r["policy_version"]) for r in rows if r.get("policy_version") is not None]
        if played and min(played) < g.version:  # a replayed group took tries an earlier leg kept: its staleness is
            self.feed.emit("group_reused", gid=g.gid, pinned=g.version, oldest=min(played),  # the oldest try's
                           reused=sum(1 for x in played if x < g.version))
            g.version = min(played)
        if mg.get("ref_fallback") and v.get("base0"):
            self.mg_v0_add(g.unit, rows)
        g.done = {"folder": str(folder), "tries": len(rows)}
        g.finished, g.status = time.time(), "ready"
        v.setdefault("latency", []).append(g.finished - g.created)
        v["rows"][g.gid] = rows
        v["running"].pop(g.gid, None)
        v["tasks"].pop(g.gid, None)
        g.tokens = sum(int(r.get("completion_tokens") or 0) for r in rows or [])
        v["buffer"].add(g)
        v["card2"].observe_group(g.finished, g.size)
        cur = getattr(self, "mg_curr", None)
        if cur is not None and rows and spec.get("role") == "train":  # as mg_curriculum_observe, per group; the
            from .rl_single import (  # decay runs once per step (v7_step), not per group
                clear_rhae,
                clear_tokens,
            )

            for grp in self.mg_groups(rows, int(cur.cfg["group_size"])).values():
                cur.observe(spec["game"], int(spec["level"]), [
                    (bool(clear_tokens(r)), clear_tokens(r), self.mg_reward(r, spec["name"]), clear_rhae(r),
                     r.get("outcome") == "budget") for r in grp])
        extra = {}
        if getattr(self, "v7e", None) is not None and spec.get("role") == "train":  # cuts, running caps (envs)
            try:
                extra = await self.v7e_group_done(g, spec, folder, rows)
            except Exception as error:  # noqa: BLE001 - a failed cut never loses the group
                self.feed.emit("alert", why="v7 envs: group harvest failed", gid=g.gid, error=repr(error)[:300])
        self.feed.emit("group_done", gid=g.gid, set=g.unit, version=g.version, seconds=round(g.finished - began, 1),
                       tries=len(rows), clears=sum(1 for r in rows if r.get("cleared")), **extra)

    def v7_unfinished(self, g):
        """A running group's share of in_flight: its planned tries, or (mg.v7.in_flight_mode "unfinished", hero-v6e)
        only those not finished yet. v6d 10:47: 12 groups mostly finished held group creation with 9 tries live."""
        if (self.v7_cfg_now.get("in_flight_mode") or "planned") != "unfinished":
            return g.size
        from . import level_bench

        return max(0, int(g.size) - int(level_bench.BENCH_DONE.get(f"{g.gid}-{g.unit}", 0)))

    async def v7_scheduler(self, sets, pool):
        from . import level_bench

        v, A = self.v7, self.v7_cfg_now
        while not v["stop"]:
            slots = level_bench.SHARED_SLOTS
            # explorer benches (floor priority, mg.explore) wait for slots no group wants: their queued tries never
            # count as server 1's queue (xv7a pfA: 4 queued explorers stopped group creation for 4 min per step)
            queued = slots.queued() if slots is not None else 0
            in_flight = sum(self.v7_unfinished(g) for g in v["running"].values())
            if v["scheduler"].want(time.time(), queued, in_flight, len(v["buffer"].ready)):
                if self.v7_make_group(sets, pool) is None:  # every unit spent or stale: keep polling (explorers and
                    now = time.time()  # finished groups add starts); one starved row a minute for the watcher
                    if now - v.get("starved_at", 0.0) >= 60.0:
                        v["starved_at"] = now
                        self.feed.emit("starved", in_flight=in_flight, queued=queued, ready=len(v["buffer"].ready),
                                       pool=len(getattr(self, "mg_harvest", None) or []))
            now = time.time()
            if now - v.get("knobs_at", 0.0) >= float(A.get("knob_poll_s", 30.0)):  # live admission knobs
                v["knobs_at"] = now
                try:
                    self.v7_admission_knobs(int(getattr(v.get("book"), "served", 0) or 0))
                except Exception as error:  # noqa: BLE001 - a bad knobs file never stops play
                    self.feed.emit("alert", why="v7 live knobs failed", error=repr(error)[:200])
            await asyncio.sleep(float(A["poll_s"]))

    # ---- trainer ----------------------------------------------------------------------------------------------

    def v7_specs(self, kept):
        """The trainer specs of one group's selected tries (as mg_update builds them)."""
        mg, specs = self.cfg["mg"], []
        level_norm = mg.get("level_norm")
        for folder, row, w, share, *rest in kept:
            spec = self.windows(folder, [row], prefix="t-")[0]
            name = rest[0] if rest else None
            extra = {"set": name}
            if level_norm and name:
                ref = (self.mg_screen["sets"].get(name, {}).get("metrics") or {}).get("median_to_clear")
                extra["norm_cap"] = float(ref or self.cfg["mg_sets"][name].get("cap") or mg["cap"])
            specs.append({**spec, "w": float(w), "whole": True, "turn_key": (str(Path(folder)), row["label"]),
                          "window": int(self.cfg["lvl_window"]), "floor": 1024, "margin": 1024, "credited_only": True,
                          "share": share, **extra})
        return specs

    def v7_trainer_args(self):
        mg = self.cfg["mg"]
        return {"signed": True, "share_prefix": True, "prefix_chunk": int(self.cfg["path_chunk"]),
                "norm": self.cfg["l2_norm"], "norm_cap": int(mg["cap"]), "batch_tokens": int(mg.get("batch_tokens") or 0),
                "batch_max": int(mg.get("batch_max") or 8), "max_windows": 100000, "clip": self.cfg["clip"],
                **dict(mg.get("trainer_fast") or {})}

    def v7_lr(self):
        """lr fixed (knob), warm-up over the first ``warmup`` steps of every start and resume (0.5x, then 1x)."""
        fixed = (self.v7_knob_file() or {}).get("lr_fixed")  # live knob: hold lr
        if fixed is not None and float(fixed) > 0:
            return float(fixed)
        lr, warm, k = float(self.cfg["lr"]), int(self.v7_cfg_now["warmup"]), int(self.v7["since_resume"])
        return lr * min(1.0, (k + 1) / warm) if warm > 0 else lr

    def v7_prepare(self, g):
        """A group's trained specs and row head (selection, specs, per-group balance; the balance spends the replies'
        turn credit). Cached per gid, so a group prepared early (``ahead``) is accumulated with the same specs."""
        mg, v = self.cfg["mg"], self.v7
        cache = v.setdefault("prepared", {})
        if g.gid in cache:
            return cache.pop(g.gid)
        rows = v["rows"].pop(g.gid, None)
        if rows is None:  # a resumed group: its rows from its folder
            rows = self.mg_rows(g.done["folder"])
        kept, info = self.mg_select(Path(g.done["folder"]), rows, g.unit)
        kept = [(*k, g.unit) if len(k) == 4 else k for k in kept]
        specs = self.v7_specs(kept)
        row = {"gid": g.gid, "set": g.unit, "version": g.version, "tries": len(rows),
               "trained": len(specs), **{k: info.get(k) for k in ("reward_mean", "reward_std", "zero_spread")}}
        if specs and mg.get("turn_credit") and mg.get("balance"):  # per group (v7): push-down = the group's push-up mass
            turn = getattr(self, "mg_turn", {})
            row["balance"] = self.mg_reply_balance(specs, turn)
            for sp in specs:
                turn.pop(tuple(sp["turn_key"]), None)
        return row, specs

    def v7_ahead(self):
        """mg.v7.ahead (MFU-1 m1k2): the specs of the group the trainer takes next (the buffer's head, if its lag
        keeps it), prepared now so the trainer encodes its first windows while the current call runs. None: nothing
        ready, or ahead off."""
        v = self.v7
        if not self.v7_cfg_now.get("ahead") or not v["buffer"].ready:
            return None
        g = v["buffer"].ready[0]
        lag = v["book"].lag(g.version)
        if lag is None or lag > v["buffer"].s_max:
            return None
        cache = v.setdefault("prepared", {})
        if g.gid not in cache:
            cache[g.gid] = self.v7_prepare(g)
        return cache[g.gid][1] or None

    async def v7_accumulate(self, g, lag, first):
        """One group into the trainer's gradient sums. Returns the group row, or None for a group without trained
        tries (no selection: a tie group)."""
        v = self.v7
        head, specs = self.v7_prepare(g)
        row = {**head, "lag": lag}
        if not specs:
            return row
        ahead = self.v7_ahead()
        want = int(self.v7_cfg_now["k3_windows"])
        per = max(1, -(-want // int(self.v7_cfg_now["batch_groups"])))
        keep = [sp["name"] for sp in specs[::max(1, len(specs) // per)][:per]]
        v["k3_specs"] += [sp for sp in specs if sp["name"] in keep]
        began = time.time()
        res = await self.trainer.call("accumulate", lr=self.v7_lr(), attempts=specs, first=first, keep_old=keep,
                                      **({"ahead": ahead} if ahead else {}), **self.v7_trainer_args())
        guard = (self.cfg.get("mg") or {}).get("window_guard")
        if not res.get("ok"):
            err = str(res.get("error") or res.get("stop") or "not finite")
            if guard and "out of memory" in err.lower():  # v7 has no in-call retry: later groups train at the fallback
                self.window_guard(v["book"].served, {"step": None, "why": "oom", "error": err[:200]})
            raise StepFailed(f"accumulate {g.gid}: {err}")
        if guard and res.get("card_peak_gib") is not None:  # mg.window_guard (MFU-1 m1w1, hero 40,960 window): a
            self.window_guard(v["book"].served, {"step": None, "torch_peak_gib": res["card_peak_gib"]})  # peak past it
        row.update({k: res.get(k) for k in ("trained_tokens", "train_seconds", "entropy")})
        # card-2 gate (MFU-1 23 Oct): the call's wall, card util, peak and host gaps per accumulate
        row["call_seconds"] = round(time.time() - began, 1)
        row.update({k: res[k] for k in ("gpu_util_mean", "card_peak_gib", "window_wait_seconds", "expand_seconds")
                    if res.get(k) is not None})
        if ahead or (res.get("encode") or {}).get("ahead_hits"):
            row["ahead"] = {"sent": len(ahead or []), "hits": (res.get("encode") or {}).get("ahead_hits")}
        if res.get("window_gap"):
            row["window_gap"] = {k: res["window_gap"].get(k) for k in ("sum", "host")}
        tis = [w.get("signed", {}).get("tis_out") for w in res.get("windows") or [] if (w.get("signed") or {}).get("tis_out")]
        if tis:
            row["tis_out"] = {"tokens": sum(t["tokens"] for t in tis), "of": sum(t["of"] for t in tis)}
        self.feed.emit("accumulate", **{k: row.get(k) for k in ("gid", "set", "lag", "trained", "trained_tokens",
                                                                 "train_seconds", "call_seconds", "gpu_util_mean",
                                                                 "card_peak_gib", "window_wait_seconds",
                                                                 "expand_seconds", "window_gap", "ahead", "reward_mean", "reward_std", "zero_spread", "balance", "tis_out") if k in row})
        return row

    async def v7_catch_up(self):
        """A version that waited for a slot is served once a release frees one."""
        v = self.v7
        version = v["book"].take_pending()
        if version is not None:
            seconds = await self.mg_serve(v["adapters"][version], version)
            self.feed.emit("served_late", version=version, load_seconds=seconds, live=v["book"].live())

    async def v7_step(self, step, groups):
        """The batch's optimizer step, the new version published, the k3 measure (alert only), the checkpoint and
        the loop state."""
        mg, v = self.cfg["mg"], self.v7
        version = v["book"].current + 1
        adapter_dir = self.work / "adapters" / f"v{version:04d}"
        lr = self.v7_lr()
        step_began = time.time()
        res = await self.trainer.call("step", lr=lr, save=str(adapter_dir), **self.v7_trainer_args())
        if not res.get("ok"):
            raise StepFailed(f"step {step}: {res.get('error') or res.get('stop')}")
        step_call_s = round(time.time() - step_began, 1)
        served = v["book"].publish(version)
        v["adapters"][version] = str(adapter_dir)
        if v["book"].pending is not None:  # no deadlock: new groups only take the served version, so older ones drain
            self.feed.emit("slot_wait", version=v["book"].pending, served=v["book"].served, live=v["book"].live(),
                           users=dict(v["book"].users))
        load_s = await self.mg_serve(str(adapter_dir), version) if served else None
        row = {"step": step, "version": version, "lr": lr, "groups": len(groups),
               "lag": lag_hist([x["lag"] for x in groups]), "tries": sum(x["tries"] for x in groups),
               "trained": sum(x["trained"] for x in groups),
               **{k: res.get(k) for k in ("grad_norm", "loss", "entropy", "trained_tokens", "train_seconds",
                                          "update_norm", "torch_peak_gib")},
               "load_seconds": load_s, "live_versions": v["book"].live(), "step_call_seconds": step_call_s}
        outs = [x["tis_out"] for x in groups if x.get("tis_out")]
        if outs:
            row["tis_out"] = {"tokens": sum(o["tokens"] for o in outs), "of": sum(o["of"] for o in outs)}
            share = row["tis_out"]["tokens"] / max(1, row["tis_out"]["of"])
            if share > float(mg.get("tis_out_alert", 0.05)):
                self.feed.emit("alert", why="TIS: token share outside [0.5, 2] over the alert line", step=step,
                               share=round(share, 4))
        if v["k3_specs"]:  # pi_old = the step's own forward (kept): the new state scored on the same windows
            k3_began = time.time()
            ahead = self.v7_ahead()  # the next batch's first group, encoded during the k3 pass
            k3 = await self.trainer.call("k3_score", attempts=v["k3_specs"],  # rl1 op_k3_score: forward only
                                         gspo_clip=list(mg.get("gspo_clip") or [3e-4, 4e-4]),
                                         **({"ahead": ahead} if ahead else {}), **self.v7_trainer_args())
            row["k3"] = {k: k3.get(k) for k in ("approx_kl", "approx_kl_tok", "k3_tok_tail", "clip_frac", "replies", "missing",
                                                 "log_s_abs_p90")}
            row["k3"].update(seconds=round(time.time() - k3_began, 1),
                             **{k: k3[k] for k in ("gpu_util_mean", "window_gap") if k3.get(k) is not None})
            if mg.get("target_kl") and (k3.get("approx_kl") or 0) > float(mg["target_kl"]):
                over = "stop" if k3["approx_kl"] > 1.5 * float(mg["target_kl"]) else "target"
                self.feed.emit("alert", why=f"k3 over {over} (alert only, step taken)", step=step,
                               approx_kl=k3["approx_kl"], lr=lr)
            v["k3_specs"] = []
        if not v["book"].slots_ok():
            self.feed.emit("alert", why="v7: live versions over max_loras", live=v["book"].live(),
                           max_loras=v["book"].max_loras)
        ckpt = self.v7_state_dir(step)
        began = time.time()
        background = bool(mg.get("async_checkpoint"))  # MFU-1 E: host snapshot now, written by a trainer thread
        res = await self.trainer.must("checkpoint", path=str(ckpt), **({"background": True} if background else {}))
        row["checkpoint_seconds"] = round(time.time() - began, 1)
        v["since_resume"] += 1
        if getattr(self, "mg_curr", None) is not None:
            self.mg_curr.decay()
        state = self.v7_state_dict(step, ckpt)
        self.v7_ckpt_flush()  # the previous background write was joined by this call: its state goes first
        if background and res.get("background"):
            row["checkpoint_background"] = True  # v7_state.json follows once s<step> lands (never a .partial)
            v["ckpt_pending"] = state
            v["ckpt_watch"] = asyncio.ensure_future(self.v7_ckpt_watch(state["checkpoint"]))
        else:
            if res.get("background_refused"):
                row["checkpoint_refused"] = res["background_refused"]
            self.v7_put_state(state)
        self.v7_prune_states(step)
        self.feed.emit("step", **row)
        return row

    def v7_ckpt_flush(self):
        """Writes the deferred loop state once its checkpoint folder exists (renamed from .partial = complete)."""
        state = self.v7.get("ckpt_pending")
        if not state or not Path(state["checkpoint"]).is_dir():
            return False
        self.v7["ckpt_pending"] = None
        self.v7_put_state(state)
        self.feed.emit("checkpoint_state", step=state["step"])
        return True

    async def v7_ckpt_watch(self, path, every=5.0):
        while (self.v7.get("ckpt_pending") or {}).get("checkpoint") == path:
            if self.v7_ckpt_flush():
                return
            await asyncio.sleep(every)

    async def v7_ckpt_join(self):
        """The run's end: the last background write lands and its state is written."""
        pending = self.v7.get("ckpt_pending")
        if not pending:
            return
        done = await self.trainer.must("checkpoint_wait", timeout=3600)
        # the watcher task may flush it during the wait (v6e S leg 24 Oct: a None pending crashed the run's end)
        landed = self.v7.get("ckpt_pending") is None or self.v7_ckpt_flush()
        if not landed:
            self.feed.emit("alert", why="v7: background checkpoint did not land", done=done, step=pending["step"])
        else:
            self.feed.emit("checkpoint_done", **{k: x for k, x in done.items() if k != "ok"})

    def v7_prune_states(self, step):
        """Keep the last 3 states, every 8th, and every open read's version."""
        root = Path(os.environ["ARC_VALIDATION_OUTPUT"]) / "state"
        keep = {int(step) - i for i in range(3)} | {int(r["version"]) for r in self.v7["reads"].open.values()}
        for p in root.glob("s[0-9][0-9][0-9][0-9]") if root.is_dir() else []:
            n = int(p.name[1:])
            if n not in keep and n % 8:
                import shutil

                shutil.rmtree(p, ignore_errors=True)

    # ---- reads -------------------------------------------------------------------------------------------------

    async def v7_read(self, step, version, sets, fixed, extra=None):
        """The held-out read pinned to ``version``: every fixed set's starts, ``read_tries`` tries per set."""
        from .rl_single import held_metrics

        mg, A, v = self.cfg["mg"], self.v7_cfg_now, self.v7
        jobs, names = [], []
        cap = None
        if float(A.get("read_share") or 0) > 0 and mg.get("active_tries"):  # the read's concurrent tries, taken out
            cap = max(1, int(float(A["read_share"]) * self.v7_slots(A)))  # of the slot pool (not added to it)
        gate = None
        if cap:  # one gate for every open read (reads overlap: a read lands ~1 h after its step); a changed cap
            gate = v.get("read_gate")  # (active_tries knob) moves the free count in place: open jobs release into it
            if gate is None:
                gate = v["read_gate"] = {"cap": cap, "free": cap, "cond": asyncio.Condition()}
            elif gate["cap"] != cap:
                async with gate["cond"]:
                    gate["free"] += cap - gate["cap"]
                    gate["cap"] = cap
                    gate["cond"].notify_all()

        progress = {"sets": 0, "tries": 0}
        self.v7_read_shares()

        async def capped(job, n):
            try:
                return await capped_run(job, n)
            finally:  # read progress on the feed (a read's fraction done is visible)
                progress["sets"] += 1
                progress["tries"] += n
                self.feed.emit("read_progress", step=step, version=version, sets_done=progress["sets"],
                               sets=len(fixed), tries_done=progress["tries"], cap=gate["cap"] if gate else None)

        async def capped_run(job, n):
            if gate is None:
                return await job
            n = min(n, gate["cap"])  # a job wider than a later, smaller cap waits for free >= its own width
            async with gate["cond"]:
                await gate["cond"].wait_for(lambda: gate["free"] >= n)
                gate["free"] -= n
            try:
                return await job
            finally:
                async with gate["cond"]:
                    gate["free"] += n
                    gate["cond"].notify_all()

        for name, held in fixed.items():
            st = held if isinstance(held, list) else [held] if held else []
            if int(A.get("read_per_start") or 0) > 0:  # envs: tries per recorded start; a fresh set its own count
                tries = int(A["read_per_start"]) if st else int(A.get("read_fresh_tries") or 2 * int(A["read_per_start"]))
            else:
                tries = max(1, int(A["read_tries"]) // max(1, len(st)))
            n = tries * max(1, len(st))
            jobs.append(capped(self.mg_play(f"x{step:04d}-fixed-{name}", sets[name], st, tries,
                                            model=self.v7_model(version), seed=int(mg.get("seed", 8000)) + 500,
                                            concurrency=min(n, cap) if cap else None,
                                            own={**(self.v7_keep() or {}), "pool_class": "read"}), n))  # kept tries: a restart reuses them
            names.append(name)
        n = len(jobs)
        v["reads"].start(step, version, n)
        v["book"].acquire(version, n, read=True)
        began = time.time()
        try:
            got = await asyncio.gather(*jobs, return_exceptions=True)
        finally:
            v["book"].release(version, n, read=True)
            await self.v7_catch_up()
        seconds = round(time.time() - began, 1)
        v.setdefault("read_seconds", []).append(seconds)
        out = {}
        for name, r in zip(names, got):
            v["reads"].try_done(step)
            if isinstance(r, BaseException):
                out[name] = {"error": repr(r)[:300]}
                continue
            out[name] = held_metrics(r[1])
        if int(version) == 0:
            v["base_read_done"] = True  # the ratio's anchor landed (a resume re-reads it until it has)
        self.feed.emit("read", step=step, version=version, sets=out, seconds=seconds, **({"base": True} if step == 0 else {}),
                       **({"extra_sets": sorted(extra)} if extra else {}))  # extended sets: reported apart from the core
        alert = mg.get("read_alert") or {}
        for name, m in out.items():
            if name in alert and (m.get("rhae_per_10k") is not None) and m["rhae_per_10k"] <= float(alert[name]):
                self.feed.emit("alert", why="held-out read at or under its line (alert only)", step=step, set=name,
                               rhae_per_10k=m["rhae_per_10k"], line=alert[name])
        return out

    def v7_knobs(self, step):
        """The live knobs file at a step boundary (rl_single.mg_knobs: play concurrency, trainer paths, explorer KV
        share, read_every / read_per_start); the read schedule follows a changed read_every."""
        mg = self.cfg["mg"]
        held = mg.get("trainer_fast")
        before = dict(mg.get("v7") or {})
        self.mg_knobs(step)
        if self.__dict__.pop("mg_tf_change", None) is not None:  # RL 23 Oct: no knob gate in v7 yet: a trainer_fast
            mg["trainer_fast"] = held  # change from the knobs file is refused, the old paths kept
            self.mg_tf_refused = (self.v7_knob_file() or {}).get("trainer_fast")
            if self.trainer is not None:
                self.trainer.always = dict(held or {})
            self.feed.emit("knobs", round=step, changed={}, refused=["trainer_fast"], why="v7: no trainer_fast gate")
        self.v7_admission_knobs(step)
        A = self.v7_cfg_now
        for key in ("read_every", "read_per_start"):  # only a value the knobs file changed (the bundle's own value
            got = (mg.get("v7") or {}).get(key)  # stays as v7_cfg set it: a preflight shrink keeps its own)
            if got is not None and got != before.get(key):
                A[key] = got
                if key == "read_every":
                    self.v7["reads"].every = int(got)

    def v7_admission_knobs(self, step):
        """Scheduler and admission knobs, read live (no restart for a settings change): the
        scheduler loop applies them every ``knob_poll_s`` and each step boundary again. ramp_s, in_flight,
        in_flight_mode, admit_stagger_s, active_kv_fill, active_tries, card2_continuous, kv_engine_scale, admit_stagger2_s, card2_idle_end_ready,
        card2_idle_min_serve_s, card2_max_block_s, card2_idle_min_s, card2_end_on_batch, s_max, active_tries2, card2_turn_cap, turn_cap, turn_auto."""
        mg, A = self.cfg["mg"], self.v7_cfg_now
        f = self.v7_knob_file()
        seen = json.dumps(f, sort_keys=True)
        if seen != getattr(self, "v7_knobs_seen", None):  # what the run read (v6h: a launcher put that failed
            self.v7_knobs_seen = seen  # silently left the bundle defaults; put_knobs.py --verify checks this row)
            self.feed.emit("knobs_file", keys=f, missing=f is None, path=os.environ.get("ARENA_KNOBS"))
        f = f or {}
        from . import level_bench
        if f.get("ramp_s") is not None and float(f["ramp_s"]) != float(self.v7["scheduler"].ramp_s):
            self.feed.emit("knobs", round=step, changed={"ramp_s": [self.v7["scheduler"].ramp_s, float(f["ramp_s"])]})
            self.v7["scheduler"].ramp_s = A["ramp_s"] = float(f["ramp_s"])
        if f.get("card2_continuous") is not None and bool(f["card2_continuous"]) != bool(self.v7["card2"].continuous):
            self.feed.emit("knobs", round=step, changed={"card2_continuous": [self.v7["card2"].continuous, bool(f["card2_continuous"])]})
            self.v7["card2"].continuous = bool(f["card2_continuous"])
        if f.get("in_flight_mode") in ("planned", "unfinished") and f["in_flight_mode"] != (A.get("in_flight_mode") or "planned"):
            self.feed.emit("knobs", round=step, changed={"in_flight_mode": [A.get("in_flight_mode") or "planned", f["in_flight_mode"]]})
            A["in_flight_mode"] = f["in_flight_mode"]
        # hero-v6d 24 Oct: the group cap (planned tries of running groups) held live tries at ~46 with card-1 KV 0.22
        if f.get("in_flight") is not None and int(f["in_flight"]) != int(self.v7["scheduler"].in_flight_target):
            self.feed.emit("knobs", round=step, changed={"in_flight": [self.v7["scheduler"].in_flight_target, int(f["in_flight"])]})
            self.v7["scheduler"].in_flight_target = A["in_flight"] = int(f["in_flight"])
        pools =[p for p in (level_bench.SHARED_SLOTS, getattr(level_bench, "SHARED_SLOTS2", None)) if p is not None]
        if f.get("admit_stagger_s") is not None and pools and float(f["admit_stagger_s"]) != float(pools[0].stagger_s):
            self.feed.emit("knobs", round=step, changed={"admit_stagger_s": [pools[0].stagger_s, float(f["admit_stagger_s"])]})
            for p in pools:  # server 2 keeps its own stagger when one is set (admit_stagger2_s)
                p.stagger_s = float(mg["admit_stagger2_s"]) if (
                    p is not pools[0] and mg.get("admit_stagger2_s") is not None) else float(f["admit_stagger_s"])
            mg["admit_stagger_s"] = float(f["admit_stagger_s"])
        if f.get("active_kv_fill") is not None and pools and 0 < float(f["active_kv_fill"]) <= 1.5 and (
                float(f["active_kv_fill"]) != float(pools[0].kv_fill or 0)):
            self.feed.emit("knobs", round=step, changed={"active_kv_fill": [pools[0].kv_fill, float(f["active_kv_fill"])]})
            pools[0].kv_fill = mg["active_kv_fill"] = float(f["active_kv_fill"])  # server 1's gate (as mg_knobs)
        c2 = self.v7.get("card2")
        for key, attr, cast in (("card2_idle_end_ready", "idle_end_ready", int), ("card2_idle_min_serve_s", "idle_min_serve_s", float),
                                ("card2_max_block_s", "max_block_s", float), ("card2_idle_min_s", "idle_min_s", float),
                                ("card2_hold", "hold", bool), ("card2_idle_ready_wait_s", "idle_ready_wait_s", float),
                                ("card2_end_on_batch", "end_on_batch", bool)):
            if c2 is not None and f.get(key) is not None and cast(f[key]) != cast(getattr(c2, attr)):
                self.feed.emit("knobs", round=step, changed={key: [getattr(c2, attr), cast(f[key])]})
                setattr(c2, attr, cast(f[key]))
        p2 = getattr(level_bench, "SHARED_SLOTS2", None)
        if p2 is not None and f.get("active_tries2") is not None and int(f["active_tries2"]) != int(p2.slots):
            self.feed.emit("knobs", round=step, changed={"active_tries2": [p2.slots, int(f["active_tries2"])]})
            p2.slots = int(f["active_tries2"])  # new tries on server 2; live ones above it finish (card2_turn_cap gates them)
        if f.get("turn_auto") is not None and bool(f["turn_auto"]) != getattr(self, "turn_auto_live", True):
            self.feed.emit("knobs", round=step, changed={"turn_auto": [getattr(self, "turn_auto_live", True),
                                                                       bool(f["turn_auto"])]})
            self.turn_auto_live = bool(f["turn_auto"])  # off: server 1's turn cap stays where turn_cap puts it
        if pools and f.get("turn_cap") is not None and int(f["turn_cap"]) != int(pools[0].turn_cap or 0):
            self.feed.emit("knobs", round=step, changed={"turn_cap": [pools[0].turn_cap, int(f["turn_cap"])]})
            pools[0].turn_cap = int(f["turn_cap"]) or None  # server 1's turn gate (mg.turn_auto moves it too)
        if p2 is not None and f.get("card2_turn_cap") is not None and int(f["card2_turn_cap"]) != int(p2.turn_cap or 0):
            self.feed.emit("knobs", round=step, changed={"card2_turn_cap": [p2.turn_cap, int(f["card2_turn_cap"])]})
            p2.turn_cap = int(f["card2_turn_cap"]) or None
        if f.get("admit_stagger2_s") is not None and getattr(level_bench, "SHARED_SLOTS2", None) is not None and (
                float(f["admit_stagger2_s"]) != float(mg.get("admit_stagger2_s", -1))):
            self.feed.emit("knobs", round=step, changed={"admit_stagger2_s": [mg.get("admit_stagger2_s"), float(f["admit_stagger2_s"])]})
            mg["admit_stagger2_s"] = level_bench.SHARED_SLOTS2.stagger_s = float(f["admit_stagger2_s"])
        if f.get("kv_engine_scale") is not None:  # engine KV per client token on the group_prefix projection
            for p in pools:  # a set knob overrides kv_project.auto_scale
                if p.project is not None:
                    p.project["auto_off"] = True
                if p.project is not None and float(p.project.get("engine_scale", 1.0)) != float(f["kv_engine_scale"]):
                    self.feed.emit("knobs", round=step, changed={"kv_engine_scale": [p.project.get("engine_scale", 1.0),
                                                                                      float(f["kv_engine_scale"])]})
                    p.project["engine_scale"] = float(f["kv_engine_scale"])
        tc = mg.get("turn_credit") or {}
        if f.get("gigpo_omega") is not None and tc.get("mode") == "gigpo" and float(f["gigpo_omega"]) != float(tc.get("omega", 1.0)):
            self.feed.emit("knobs", round=step, changed={"gigpo_omega": [tc.get("omega", 1.0), float(f["gigpo_omega"])]})
            tc["omega"] = float(f["gigpo_omega"])  # audit 24 Oct: the step term's weight, live (cap if it dominates)
        curr = getattr(self, "mg_curr", None)  # curriculum settings, live (retire_p)
        for key, got in ((f.get("curriculum") or {}) if curr is not None else {}).items():
            if key in CURRICULUM_LIVE and got is not None and float(got) != float(curr.cfg.get(key, -1)):
                self.feed.emit("knobs", round=step, changed={f"curriculum.{key}": [curr.cfg.get(key), float(got)]})
                curr.cfg[key] = float(got)
        if f.get("active_tries") is not None and level_bench.SHARED_SLOTS is not None and (
                int(f["active_tries"]) != int(level_bench.SHARED_SLOTS.slots)):
            self.feed.emit("knobs", round=step, changed={"active_tries": [level_bench.SHARED_SLOTS.slots, int(f["active_tries"])]})
            level_bench.SHARED_SLOTS.slots = mg["active_tries"] = int(f["active_tries"])
        if f.get("read_share") is not None and 0 < float(f["read_share"]) <= 1 and (
                float(f["read_share"]) != float(A.get("read_share") or 0)):  # v6g's base read
            self.feed.emit("knobs", round=step, changed={"read_share": [A.get("read_share"), float(f["read_share"])]})
            A["read_share"] = float(f["read_share"])  # held 8 tries for 2 h; open reads take the new width at once
        if f.get("max_ready") is not None and int(f["max_ready"]) >= 1 and int(f["max_ready"]) != int(A["max_ready"]):
            # v6j resume: 7 fully reused replays filled max_ready 8 and held group creation for 7 min
            self.feed.emit("knobs", round=step, changed={"max_ready": [A["max_ready"], int(f["max_ready"])]})
            A["max_ready"] = int(f["max_ready"])
            if (getattr(self, "v7", None) or {}).get("scheduler") is not None:
                self.v7["scheduler"].max_ready = A["max_ready"]
        if f.get("s_max") is not None and int(f["s_max"]) >= 1 and int(f["s_max"]) != int(A["s_max"]):
            self.feed.emit("knobs", round=step, changed={"s_max": [A["s_max"], int(f["s_max"])]})
            A["s_max"] = int(f["s_max"])
            if (getattr(self, "v7", None) or {}).get("buffer") is not None:
                self.v7["buffer"].s_max = A["s_max"]
        if hasattr(self, "v7_read_gate_resize"):  # (test hosts borrow this method alone)
            self.v7_read_gate_resize()

    def v7_read_shares(self):
        """v7.read_kv_share (v6h 24 Oct): reads (pool class "read") hold at most read_share of each pool's projected-KV
        gate, so training groups keep the rest; off: reads outrank groups (oldest_first) up to the whole gate."""
        from . import level_bench

        A = self.v7_cfg_now
        for pool in (level_bench.SHARED_SLOTS, getattr(level_bench, "SHARED_SLOTS2", None)):
            if pool is not None and hasattr(pool, "shares"):  # (test fakes carry no shares)
                if A.get("read_kv_share") and float(A.get("read_share") or 0) > 0:
                    pool.shares["read"] = float(A["read_share"])
                else:
                    pool.shares.pop("read", None)

    def v7_read_gate_resize(self):
        """The open reads' gate follows read_share x active_tries (knobs) without waiting for the next read."""
        if hasattr(self, "v7_read_shares"):
            self.v7_read_shares()
        gate, A = self.v7.get("read_gate"), self.v7_cfg_now
        if gate is None or not float(A.get("read_share") or 0) > 0 or not self.cfg["mg"].get("active_tries"):
            return
        cap = max(1, int(float(A["read_share"]) * self.v7_slots(A)))
        if cap == gate["cap"]:
            return

        async def resize():
            async with gate["cond"]:
                gate["free"] += cap - gate["cap"]
                gate["cap"] = cap
                gate["cond"].notify_all()
        gate["cap_next"] = cap
        asyncio.ensure_future(resize())

    @staticmethod
    def v7_knob_file():
        path = os.environ.get("ARENA_KNOBS")
        try:
            return json.loads(Path(path).read_text()) if path and Path(path).is_file() else None
        except ValueError:
            return None

    # ---- the loop ----------------------------------------------------------------------------------------------

    def v7_v0_scan(self):
        """mg.ref_fallback: version-0 clears the lineage's legs kept (partial/v7/<gid>-<set>/tries.jsonl)."""
        own = self.v7_keep() or {}
        roots = [own.get("try_keep", {}).get("keep")] + list(own.get("try_keep", {}).get("reuse") or [])
        n = 0
        for root in [Path(r) for r in roots if r]:
            for f in sorted(root.glob("*/tries.jsonl")) if root.is_dir() else []:
                name = f.parent.name.split("-", 1)[1] if "-" in f.parent.name else None
                if name not in self.cfg["mg_sets"]:
                    continue
                rows = []
                for line in f.read_text().splitlines():
                    try:
                        rows.append(json.loads(line))
                    except ValueError:
                        continue
                self.mg_v0_add(name, rows)
                n += 1
        return n

    def v7_v0_emit(self, n):
        refs = getattr(self, "mg_v0_refs", None) or {}
        self.feed.emit("ref_fallback", mode=self.cfg["mg"]["ref_fallback"], groups=n, units=len(refs),
                       usable=sum(1 for x in refs.values() if len(x) >= 2))

    async def v7_setup(self, probes=None):
        """v6h: the v0 scan read 120 kept groups over the volume on the event loop (225 s loop lag before the first
        group): it runs on a thread beside the trainer/state load and lands before play starts."""
        if hasattr(self, "serving_log_mirror"):  # v6j: the mirror sat on the v1 round path only, so v7 never started it
            self.serving_log_mirror()
        scan = asyncio.ensure_future(asyncio.to_thread(self.v7_v0_scan)) if self.cfg["mg"].get("ref_fallback") else None
        try:
            return await self.v7_setup_body(probes)
        finally:
            if scan is not None:
                self.v7_v0_emit(await scan)

    async def v7_setup_body(self, probes=None):
        mg, A = self.cfg["mg"], self.v7_cfg_now
        state = self.v7_resume_state()
        boot, self.mg_trainer_boot = getattr(self, "mg_trainer_boot", None), None
        if boot is not None:  # mg.timeshare.lazy_boot: mg_loop started the trainer at its top
            await boot
        if getattr(self, "trainer", None) is None:
            await self.start_trainer()
        if mg.get("reload_check"):  # the probes' base references, before any state loads (the reload check
            # compares kl_base; v6f2 died at its first step on a check with none)
            if not probes:
                raise ValueError("mg.reload_check needs probe windows")
            await self.mg_base_reference(probes)
        self.v7 = {"rows": {}, "tasks": {}, "running": {}, "k3_specs": [], "stop": False, "adapters": {},
                   "made": {"n": 0, "by_set": {}}, "since_resume": 0, "base0": True,
                   "scheduler": Scheduler(A["group_size"], A["max_ready"], A["ramp_s"], A["in_flight"]),
                   "card2": Card2(**{"on": False, **(A["card2"] or {})}), "reads": Reads(A["read_every"])}
        v = self.v7
        if state is None:
            v["book"], v["buffer"], step = (Book(A["max_loras"], A["serve_every"], read_slots=int(A.get("read_slots", 1))),
                                            Buffer(A["s_max"], A["batch_groups"]), 0)
            init = mg.get("init_state")
            if not init and mg.get("resume") and (Path(mg["resume"]) / "mg_state.json").is_file():
                init = json.loads((Path(mg["resume"]) / "mg_state.json").read_text()).get("checkpoint")  # a v6x state
            if init:  # version 0 = that state's adapter (and optimizer), served under its versioned name
                await self.trainer.must("load_checkpoint", path=init)
                path = self.work / "adapters" / "v0000"
                await self.trainer.must("save", path=str(path))
                v["adapters"][0], v["base0"] = str(path), False
                await self.mg_serve(str(path), 0)
            v["book"].free = {0} if v["base0"] else set()  # the base model needs no LoRA slot
            if self.v7e_on():
                await self.v7e_setup(None)
            self.feed.emit("v7_start", resume=False, init=init, cfg=A)
            return step
        await self.trainer.must("load_checkpoint", path=self.v7_found(state["checkpoint"]))
        # every setting is the leg's, the saved state keeps only evidence (v6j: max_loras and s_max were the saved leg's)
        v["book"], v["buffer"] = (Book.load_state(state["book"], read_slots=int(A.get("read_slots", 1)),
                                                  max_loras=int(A["max_loras"]), serve_every=int(A["serve_every"])),
                                  Buffer.load_state(state["buffer"], s_max=int(A["s_max"]),
                                                    batch_groups=int(A["batch_groups"])))
        lost_ready = []
        for g in list(v["buffer"].ready):  # finished groups train from their folder (the earlier container's path);
            # one whose tries never reached the volume (a preempted container syncs only feed/status) is replayed
            g.done["folder"] = self.v7_found(g.done.get("folder") or "")
            if not (Path(g.done["folder"]) / "level-bench.json").is_file():
                v["buffer"].ready.remove(g)
                lost_ready.append({"gid": g.gid, "unit": g.unit, "start": g.start, "version": g.version, "size": g.size,
                                   "created": g.created})
        v["reads"] = Reads.load_state(state["reads"], every=int(A["read_every"]))
        v["card2"].load(state.get("card2") or {})
        v["made"] = state["made"]
        v["base_read_done"] = bool(state.get("base_read_done"))
        # adapters lived on the dead container's local disk: only the checkpoint's version (the current one) is
        # rebuilt (trainer save of the loaded state, as the v6 resume); new and replayed groups all play on it
        v["base0"] = "0" not in state["adapters"] and 0 not in state["adapters"]
        book, cur = v["book"], v["book"].current
        v["adapters"] = {}
        if cur == 0 and v["base0"]:
            pass  # still the base model: nothing to serve
        else:
            path = self.work / "adapters" / f"v{cur:04d}"
            await self.trainer.must("save", path=str(path))
            v["adapters"][cur] = str(path)
            await self.mg_serve(str(path), cur)
        book.served, book.pending, book.users, book.read_users = cur, None, {}, {}
        book.free = {0} if v["base0"] else set()  # the base model needs no LoRA slot
        cur_curr = getattr(self, "mg_curr", None)
        if cur_curr is not None and state.get("curriculum"):
            self.mg_curr = type(cur_curr).load_state(state["curriculum"])
            # a code restart's curriculum settings are the bundle's (hero-v6c: chain_* on a v6b state); evidence stays
            changed = {k: [self.mg_curr.cfg.get(k), v] for k, v in cur_curr.cfg.items() if self.mg_curr.cfg.get(k) != v}
            if changed:
                self.mg_curr.cfg.update(cur_curr.cfg)
                self.feed.emit("curriculum_cfg", changed=changed)
        if self.v7e_on():  # a state written before the envs keys (v7k/r12): a fresh pool beside the restored rest
            await self.v7e_setup(state.get("envs"))
        v["reads_lost"] = sorted(v["reads"].open.items())  # a read cut by the restart: re-read on the resumed state
        v["reads"].open = {}
        replay = list(state["running"]) + lost_ready
        # v6h: a replayed group reuses its kept tries, so it keeps their (oldest) version; one already past s_max at
        # the resume is dropped before any try is replayed (v6h played 6 such groups' missing tries, then dropped all 18)
        stale = [g for g in replay if cur - int(g["version"]) > int(A["s_max"])]
        replay = [g for g in replay if cur - int(g["version"]) <= int(A["s_max"])]
        for g in stale:
            v["buffer"].closed[g["gid"]] = "dropped"
            d = {"gid": g["gid"], "unit": g["unit"], "version": int(g["version"]), "lag": cur - int(g["version"]),
                 "at": time.time(), "trainer_version": cur, "why": "resume_lag", "_seen": True}
            v["buffer"].drops.append(d)
            self.feed.emit("group_drop", **{k: x for k, x in d.items() if k != "_seen"})
        for g in replay:  # unfinished (and lost finished) groups: replayed whole, re-pinned to the resumed version
            grp = Group(**{**g, "version": cur, "done": {}, "finished": None, "status": "running", "created": time.time()})
            if not grp.cap:  # saved before caps were pinned: the cap its kept tries played at, so their digests match
                grp.cap = self.v7_kept_cap(grp)
            v["running"][grp.gid] = grp
            v["tasks"][grp.gid] = asyncio.ensure_future(self.v7_play_group(grp, self.v7_spec(grp)))
        self.feed.emit("v7_start", resume=True, step=state["step"], version=cur, buffered=len(v["buffer"].ready),
                       live={"max_loras": v["book"].max_loras, "read_slots": v["book"].read_slots, "s_max": v["buffer"].s_max,
                             "batch_groups": v["buffer"].batch_groups, "serve_every": v["book"].serve_every,
                             "read_every": v["reads"].every},  # the values in force (not the saved state's)
                       replayed=len(replay), dropped_stale=len(stale), lost_ready=len(lost_ready), reads_lost=[k for k, _ in v["reads_lost"]],
                       cfg=A, repinned={g["gid"]: g["version"] for g in replay})  # lost work: the original versions
        return int(state["step"])

    async def mg_async_run(self, *, sets, pool, fixed, tests, probes, screen):
        from . import level_bench

        self.v7_cfg_now = A = self.v7_cfg()
        mg = self.cfg["mg"]
        if mg.get("active_tries"):  # as mg_loop (mg5f) past the async hook: one KV-aware attempt-slot pool; server
            # 2's pool joins it only through this one (level_bench join2), so without it Mode B moves no try
            level_bench.SHARED_SLOTS = level_bench.SharedSlots(self.v7_slots(A), mg.get("active_kv_fill"),
                                                                  project=mg.get("kv_project"),
                                                                  stagger_s=mg.get("admit_stagger_s") or 0.0)
            if mg.get("engine_admission") and getattr(self, "mg_admission_task", None) is None:
                self.mg_admission_task = asyncio.ensure_future(self.mg_engine_admission())
        elif (A.get("card2") or {}).get("on"):
            raise ValueError("mg.v7.card2 needs mg.active_tries (server 2 shares its slot pool)")
        step = await self.v7_setup(probes)
        v = self.v7
        if getattr(self, "served_dir", None):
            self.served_persist_start()
            await self.served_restored()
        sched = asyncio.ensure_future(self.v7_scheduler(sets, pool))
        reads, batch, why = [], [], "max_steps"
        if step == 0 and fixed and A.get("read_base") and v["book"].current == 0:  # version 0: the base's own read
            reads.append(asyncio.ensure_future(self.v7_read(0, 0, sets, fixed)))
        wread = getattr(self, "v7e", None) is not None and self.cfg["mg"].get("wread")
        if wread and self.cfg["mg"]["wread"].get("base", True) and not self.v7e.get("wread_base") and v["base0"] and (
                step == 0 or v["book"].current > 0):  # the base's whole-game read (a resume onto wread: once, base model)
            reads.append(asyncio.ensure_future(self.v7e_wread(0, 0)))
        if (step > 0 and fixed and A.get("read_base") and v["base0"] and not v.get("base_read_done")
                and "0" not in [str(k) for k, _ in v.get("reads_lost") or []]):  # v6e: a lineage whose base read
            self.feed.emit("base_reread", step=step)  # never landed (v6d cut it, v6e re-read on version 1) reads it
            reads.append(asyncio.ensure_future(self.v7_read(0, 0, sets, fixed)))
        if v.get("reads_lost") and fixed:  # a read the restart cut: one read of the resumed state, at its step
            lost = [str(k) for k, _ in v["reads_lost"]]
            base = "0" in lost and A.get("read_base") and v["base0"] and not v.get("base_read_done")  # the base's read is the ratio's anchor: re-read
            self.feed.emit("read_lost", steps=lost, reread_step=step, base_reread=bool(base))  # on version 0 (v6e)
            if base:
                reads.append(asyncio.ensure_future(self.v7_read(0, 0, sets, fixed)))
            if [k for k in lost if k != "0"] or not base:
                reads.append(asyncio.ensure_future(self.v7_read(step, v["book"].served, sets, fixed)))
        try:
            for _ in range(3):  # card2.start_block: card 2 plays while the first groups are in play (a resume: if idle)
                if batch or not await self.v7_maybe_block(step, probes):
                    break
            while step < int(A["max_steps"]):
                if sched.done() and sched.exception() is not None:
                    raise sched.exception()
                g, lag = v["buffer"].next_group(v["book"], time.time())
                for d in [d for d in v["buffer"].drops if d.get("_seen") is None]:  # every drop (v6h: one
                    d["_seen"] = True  # next_group call dropped 12 replayed groups, only the last was on the feed)
                    v.get("prepared", {}).pop(d["gid"], None)  # prepared early (ahead), then dropped on lag
                    self.feed.emit("group_drop", **{k: x for k, x in d.items() if k != "_seen"})
                if g is None:
                    if await self.v7_idle_block(step, probes, batch):
                        continue
                    if not v["buffer"].ready:
                        v.setdefault("bal", {"wait_s": 0.0, "serve_s": 0.0, "blocks": 0})["wait_s"] += float(A["poll_s"])
                    await asyncio.sleep(float(A["poll_s"]))
                    continue
                v.pop("wait_since", None)
                try:
                    row = await self.v7_accumulate(g, lag, first=not batch)
                except StepFailed as error:  # a poisoned batch: reset, its groups never train
                    await self.trainer.call("acc_reset")
                    for x in batch + [{"gid": g.gid}]:
                        v["buffer"].closed[x["gid"]] = "failed"
                    self.feed.emit("alert", why="v7: accumulate failed; batch dropped", error=str(error)[:300],
                                   groups=[x["gid"] for x in batch] + [g.gid])
                    batch, v["k3_specs"] = [], []
                    continue
                v["buffer"].close(g, "consumed")
                batch.append(row)
                if len(batch) < int(A["batch_groups"]):
                    continue
                if not any(x["trained"] for x in batch):  # all ties: nothing to step on
                    self.feed.emit("step_skip", groups=[x["gid"] for x in batch])
                    batch = []
                    continue
                step += 1
                began = time.time()
                row = await self.v7_step(step, batch)
                if mg.get("reload_check") and v["since_resume"] == 1 and not self.v7_reload_known(step):
                    # the trainer key's GPU smoke (launch gate): the first step's checkpoint (start or resume) reloads
                    # and rescores the probes within the measured repeat floor, or the run stops (mg_reload_check)
                    await self.v7_ckpt_join()
                    await self.mg_reload_check(step, {}, probes, state=self.v7_state_dir(step))
                    self.v7_reload_mark(step)
                v["card2"].observe_step(row["tries"], max(1.0, time.time() - began + sum(x.get("train_seconds") or 0 for x in batch)))
                batch = []
                if getattr(self, "v7e", None) is not None:
                    self.v7e_step_row(step)
                bal = v.pop("bal", None) or {"wait_s": 0.0, "serve_s": 0.0, "blocks": 0}
                self.feed.emit("balance", step=step, trainer_wait_s=round(bal["wait_s"], 1),  # both cards
                               card2_serve_s=round(bal["serve_s"], 1), idle_blocks=bal["blocks"],  # busy, balanced
                               ready=len(v["buffer"].ready), running=len(v["running"]))
                self.v7_knobs(step)
                whole = wread and self.v7e_wread_due(step)
                if v["reads"].due(step) and fixed:  # the extended per-level sets only with a whole-game read
                    extra = self.v7e_wread_extra(fixed) if whole else {}
                    reads.append(asyncio.ensure_future(self.v7_read(step, v["book"].served, sets, {**fixed, **extra},
                                                                    extra=list(extra))))
                if whole:
                    reads.append(asyncio.ensure_future(self.v7e_wread(step, v["book"].served)))
                manual = os.environ.get("ARENA_STOP_REQUEST")  # the volume stop request: stop at this step boundary
                if manual and Path(manual).exists():
                    why = "manual"
                    break
                await self.v7_maybe_block(step, probes)
        finally:
            v["stop"] = True
            if getattr(self, "v7e", None) is not None:
                await self.v7e_stop()
            sched.cancel()
            for t in list(v["tasks"].values()):
                t.cancel()
            await asyncio.gather(sched, *v["tasks"].values(), return_exceptions=True)
        if reads and why == "manual":  # a stop request ends now: open reads are cut (hero-v2 CPU legs waited 45+ min
            # on them); the state keeps them open, so a resume re-reads them (reads_lost)
            open_reads = [t for t in reads if not t.done()]
            for t in open_reads:
                t.cancel()
            await asyncio.gather(*reads, return_exceptions=True)
            if open_reads:
                self.feed.emit("reads_cut", n=len(open_reads), open=sorted(v["reads"].open))
        elif reads:  # the last reads land before the run ends (their versions are kept), under a crash ceiling
            A = self.v7_cfg_now
            measured = v.get("read_seconds") or []
            limit = max(float(A.get("final_read_floor_s") or 0.0),
                        3.0 * max(measured) if measured else float(A.get("final_read_s") or 4500.0))
            done, pending = await asyncio.wait(reads, timeout=limit)
            if pending:  # cut; the state keeps them open, so a resume re-reads them (reads_lost)
                for t in pending:
                    t.cancel()
                await asyncio.gather(*pending, return_exceptions=True)
                self.feed.emit("reads_deadline", n=len(pending), open=sorted(v["reads"].open), limit_s=round(limit, 1),
                               slowest_s=max(measured) if measured else None)
                self.feed.emit("alert", why=f"v7 final reads past their {limit:.0f} s ceiling: {len(pending)} cut, kept "
                                            "for a resume", n=len(pending))
        await self.v7_ckpt_join()
        cur = v["book"].current
        if cur in v["adapters"] and os.environ.get("ARC_VALIDATION_OUTPUT") and Path(v["adapters"][cur]).is_dir():
            dest = Path(os.environ["ARC_VALIDATION_OUTPUT"]) / "adapters" / f"v{cur:04d}"
            await asyncio.to_thread(shutil.copytree, v["adapters"][cur], dest, dirs_exist_ok=True)  # the last version
            self.feed.emit("mg_adapter", path=str(dest), version=cur)
        if v.get("route2") is not None:  # no further block: held server-2 requests move to server 1
            v["route2"].release()
        self.feed.emit("v7_end", why=why, step=step, version=v["book"].current, drops=len(v["buffer"].drops))
        return why



def s2_progress(text):
    """(running requests, prompt + generated tokens so far) from a vLLM /metrics page, label sets summed."""
    keys = {"vllm:num_requests_running": 0, "vllm:generation_tokens_total": 1, "vllm:prompt_tokens_total": 1}
    out = [0.0, 0.0]
    for line in text.splitlines():
        name = line.split("{", 1)[0].split(" ", 1)[0]
        if not line.startswith("#") and name in keys:
            try:
                out[keys[name]] += float(line.rsplit(" ", 1)[1])
            except ValueError:
                pass
    return out[0], out[1]


class S2Health:
    """timeshare.health_*: server 2 is unhealthy after ``misses`` probes in a row that failed, timed out, or saw
    running requests with no prompt or generated token since the probe before (v6h: a live process hung in
    sample_tokens; /health alone stays 200 on a hung engine core)."""

    def __init__(self, misses=2):
        self.misses, self.row, self.last = int(misses), 0, None

    def update(self, sample):
        """``sample``: (running, tokens) or an error string. Returns the reason once ``misses`` are reached."""
        if isinstance(sample, str):
            why = sample
        else:
            running, tokens = sample
            why = (f"{running:.0f} running, no token in a probe interval"
                   if running > 0 and self.last is not None and tokens <= self.last else None)
            self.last = tokens
        self.row = self.row + 1 if why else 0
        return why if self.row >= self.misses else None


class Route2(httpx.AsyncBaseTransport):
    """httpx transport for server 2's bench clients (level_bench.TRANSPORT2): while ``redirect`` is on (server 2
    asleep after a Mode B block), every request goes to server 1 instead, so a try that started on server 2 goes on
    there (one re-prefill; server 1 serves every live version)."""

    def __init__(self, server1_url, gate=None):
        self.inner = httpx.AsyncHTTPTransport()
        self.server1 = httpx.URL(server1_url)
        self.gate = gate or (lambda: None)  # server 1's KV gate (its SharedSlots gate): moved requests queue on it
        self.redirect = False
        self.moved = 0
        self.waited = 0.0
        # hold (card2.hold): while redirected, a request waits for server 2's next block instead of moving to server 1
        self.hold, self.held, self.held_models, self.open = False, 0, set(), None
        self.hold_max_s, self.hold_timeouts = 2400.0, 0

    def reopen(self):
        """Server 2 serves again: held requests go on to it."""
        self.redirect = False
        if self.open is not None:
            self.open.set()
        self.open = None

    def close(self, hold):
        """Server 2 sleeps: later requests are held (``hold``) or moved to server 1."""
        self.hold, self.redirect = bool(hold), True

    def release(self):
        """No further block (loop end, Mode B off): held requests move to server 1 through its gate."""
        self.hold = False
        if self.open is not None:
            self.open.set()
        self.open = None

    @staticmethod
    def estimate(request):
        """Request tokens for the gate: prompt ~ body bytes / 3 plus the reply cap (max_tokens)."""
        body = request.content or b""
        try:
            cap = int(json.loads(body).get("max_tokens") or 0)
        except (ValueError, AttributeError):
            cap = 0
        return len(body) // 3 + cap

    async def handle_async_request(self, request):
        while self.redirect and self.hold:
            if self.open is None:
                self.open = asyncio.Event()
            try:
                self.held_models.add(str(json.loads(request.content or b"{}").get("model") or ""))
            except (ValueError, AttributeError):
                pass
            self.held += 1
            try:  # a ceiling: a hold past hold_max_s (no block came) moves to server 1 through its gate
                await asyncio.wait_for(asyncio.shield(self.open.wait()), timeout=self.hold_max_s)
            except asyncio.TimeoutError:
                self.hold_timeouts += 1
                break
        if not self.redirect:
            return await self.inner.handle_async_request(request)
        request.url = request.url.copy_with(scheme=self.server1.scheme, host=self.server1.host, port=self.server1.port)
        request.headers["host"] = f"{self.server1.host}:{self.server1.port}" if self.server1.port else self.server1.host
        self.moved += 1
        gate = self.gate()
        if gate is None:
            return await self.inner.handle_async_request(request)
        began = time.time()  # a moved try enters server 1 through its KV gate, never past it
        reservation = await gate.acquire(f"moved-{id(request)}", self.estimate(request))
        self.waited += time.time() - began
        try:
            response = await self.inner.handle_async_request(request)
            await response.aread()  # the whole reply inside the reservation (non-streaming requests)
            return response
        finally:
            gate.release(reservation)

    async def aclose(self):
        await self.inner.aclose()


class ModeBMixin:
    """Mode B (rl-plan v7, >= 90% util on both cards): at a step boundary, when the trainer would idle,
    the trainer parks and server 2 plays a block on card 2; at the block's end server 2's tries move to server 1."""

    def v7_group_etas(self, now):
        """(expected finish, tries) per running group from the mean group latency so far (45 min before any)."""
        lat = self.v7.setdefault("latency", [])
        mean = sum(lat[-32:]) / len(lat[-32:]) if lat else 2700.0
        return [(max(now, g.created + mean), g.size) for g in self.v7["running"].values()]

    async def v7_maybe_block(self, step, probes):
        v = self.v7
        if not v["card2"].on or getattr(self, "ts_dead", False):
            return None
        now = time.time()
        if len(v["buffer"].ready) >= int(self.v7_cfg_now["max_ready"]):  # a full buffer: the trainer has work, and
            return None  # no group could be made for server 2 during a block (xv7b pfA: 6 blocks served 0 tries)
        ready = sum(g.size for g in v["buffer"].ready)
        mode, seconds = v["card2"].decide(now, ready, self.v7_group_etas(now))
        if mode != "play":
            return None
        return await self.v7_block(step, seconds, probes)

    async def v7_idle_block(self, step, probes, batch):
        """card2.continuous: a serve block once the trainer has waited idle_min_s with nothing ready (mid-batch too); it
        ends when a group is ready. The batch's gradients ride the park (keep_grads) and are
        checked both ways; a mismatch drops the batch and turns continuous off."""
        v = self.v7
        c2 = v["card2"]
        if not c2.continuous or not c2.on or getattr(self, "ts_dead", False):
            return False
        now = time.time()
        since = v.setdefault("wait_since", now)
        seconds = c2.idle(len(v["buffer"].ready), now - since)
        if seconds <= 0:
            return False
        v.pop("wait_since", None)
        need = max(1, int(self.v7_cfg_now["batch_groups"]) - len(batch)) if getattr(c2, "end_on_batch", False) else None
        row = await self.v7_block(step, seconds, probes, end_ready=max(1, int(c2.idle_end_ready)), keep_grads=bool(batch),
                                  idle=True, batch_need=need)
        if not row or row.get("skipped"):
            return False
        bal = v.setdefault("bal", {"wait_s": 0.0, "serve_s": 0.0, "blocks": 0})
        bal["serve_s"] += float(row.get("seconds") or 0.0)
        bal["blocks"] += 1
        if batch:
            kept = (row.get("park") or {}).get("grads_kept") or {}
            back = (row.get("unpark") or {}).get("grads_restored") or {}
            a, b = float(kept.get("sq") or 0.0), float(back.get("sq") or 0.0)
            # neither side: the batch so far holds no gradients (v6fR: its one group lost both tries to the digest gate)
            if (kept or back) and (not kept or not back or abs(a - b) > 1e-6 * max(abs(a), 1e-30)):
                await self.trainer.call("acc_reset")
                for x in batch:
                    v["buffer"].closed[x["gid"]] = "failed"
                self.feed.emit("alert", why="v7 card2.continuous: gradients changed over a mid-batch park; batch "
                               "dropped, continuous off", kept=kept, restored=back, groups=[x["gid"] for x in batch])
                batch.clear()
                c2.continuous = False
        return True

    async def v7_server2_probe(self, timeout):
        """(running, tokens) from server 2's /metrics, or why the probe failed."""
        try:
            async with httpx.AsyncClient(trust_env=False, timeout=timeout) as client:
                response = await client.get(self.server2.root + "/metrics")
            if response.status_code != 200:
                return f"/metrics status {response.status_code}"
            return s2_progress(response.text)
        except Exception as error:  # noqa: BLE001 - a timeout or refused connection is a miss
            return f"/metrics {type(error).__name__}"

    def v7_keep_server2_log(self, failures):
        """The failed server 2's vllm.log tail, kept beside the feed (its folder is container-local)."""
        try:
            tail = (Path(self.server2.folder) / "vllm.log").read_text(errors="replace")[-20000:]
            out = self.output / "rl" / f"server2-failure-{failures}.log"
            out.parent.mkdir(parents=True, exist_ok=True)
            out.write_text(tail)
            return str(out)
        except Exception as error:  # noqa: BLE001 - diagnosis only
            return repr(error)[:200]

    async def v7_block(self, step, seconds, probes, end_ready=None, keep_grads=False, idle=False, batch_need=None):
        from . import level_bench

        v, mg = self.v7, self.cfg["mg"]
        row = {"step": step, "planned_s": round(seconds, 1), **({"idle": True} if idle else {})}
        began = time.time()
        if not v.get("park_checked") and (mg.get("timeshare") or {}).get("park_check", True):
            v["park_checked"] = True
            check = await self.ts_park_check(probes)
            if not (check.get("ok", True) and check.get("identical")):
                v["card2"].on = False
                return {**row, "skipped": "park check failed"}
        row["park"] = await self.trainer.must("park", base=True, **({"keep_grads": True} if keep_grads else {}))
        try:
            if getattr(self, "server2", None) is None:
                row["boot_s"] = (await self.ts_boot(sleep=False)).get("boot_seconds")
            else:
                row["wake_s"] = await self.server2.wake()
            route = v.get("route2")
            if route is None:
                route = v["route2"] = Route2(self.server.root, gate=lambda: getattr(level_bench.SHARED_SLOTS, "gate", None))
                level_bench.TRANSPORT2 = route
            served = v["book"].served
            level_bench.SERVER2_VERSIONS = {served}  # server 2 holds the newest served version (+ held tries' versions)
            if not (served == 0 and v["base0"]):
                row["load_s"] = await self.server2.serve(self.mg_name(served), v["adapters"][served])
            held = {ver for ver in v["adapters"] if self.mg_name(ver) in route.held_models and ver != served}
            for ver in sorted(held):  # held tries resume on their own (pinned) version: exact behaviour policy
                await self.server2.load(self.mg_name(ver), v["adapters"][ver])
            level_bench.SERVER2_VERSIONS |= held
            if (mg.get("timeshare") or {}).get("all_versions"):  # v6i/v6j: 40-90% of running groups at block open were
                # on an older version, so their tries could only go to server 1 (server 2 took 2-7 new tries a block)
                book, room = v["book"], int(v["book"].max_loras) - 1 - len(held)
                extra = [ver for ver in sorted(book.live(), reverse=True)
                         if ver != served and ver not in held and v["adapters"].get(ver) is not None
                         and not (ver == 0 and v["base0"])][:max(0, room)]
                for ver in extra:  # each try stays on its pinned version: exact behaviour policy
                    await self.server2.load(self.mg_name(ver), v["adapters"][ver])
                level_bench.SERVER2_VERSIONS |= set(extra)
                row["extra_versions"] = extra
            if getattr(self.server2, "keeps_lora", lambda: False)():  # mg.timeshare.keep_lora: loads were skipped
                row["pruned"] = await self.server2.prune({self.mg_name(x) for x in level_bench.SERVER2_VERSIONS})
                row["skipped_loads"] = self.server2.skipped_loads
            row["held"], row["held_versions"] = route.held, sorted(held)
            route.held, route.held_models = 0, set()
            route.reopen()
            pool2 = level_bench.SHARED_SLOTS2
            pool2.active, pool2.served = True, 0
            if mg.get("admit_stagger2_s") is not None:  # server 2's own stagger: it fills at block open
                pool2.stagger_s = float(mg["admit_stagger2_s"])
            v["scheduler"].queue_groups = 2
            self.feed.emit("block_on", **row)
            opened = time.time()
            deadline = opened + seconds
            end_ready = int(end_ready if end_ready is not None else (getattr(v["card2"], "end_ready", 0) or 0))
            floor = float(getattr(v["card2"], "idle_min_serve_s", 0.0) or 0.0) if idle else 0.0
            wait1 = float(getattr(v["card2"], "idle_ready_wait_s", 0.0) or 0.0) if idle else 0.0
            first_ready = None
            ts = mg.get("timeshare") or {}
            every = float(ts.get("health_every_s", 0.0))  # off by default; 30 with max_failures 3
            health = S2Health(ts.get("health_misses", 2)) if every > 0 and getattr(self.server2, "root", None) else None
            probed = time.time()
            while time.time() < deadline:
                await asyncio.sleep(max(0.0, min(float(self.v7_cfg_now.get("poll_s") or 5.0), deadline - time.time())))
                proc = getattr(self.server2, "process", None)  # v6i: a dead server 2 failed every try sent to it
                if proc is not None and proc.poll() is not None:  # for 2.5 min until the block ended on "ready" groups
                    raise RuntimeError(f"server 2 exited {proc.returncode} mid-block")
                if health is not None and time.time() - probed >= every:
                    probed = time.time()
                    why = health.update(await self.v7_server2_probe(float(ts.get("health_timeout_s", 20.0))))
                    if why:
                        raise RuntimeError(f"server 2 unhealthy: {why}")
                ready = len(v["buffer"].ready)
                first_ready = (first_ready or time.time()) if ready else None
                if batch_need and ready >= batch_need:  # card2.end_on_batch: these complete the batch (no floor)
                    row["ended"] = f"{ready} groups ready complete the batch"
                    break
                if end_ready and ready >= end_ready and time.time() - opened >= floor:
                    row["ended"] = f"{ready} groups ready"
                    break
                if wait1 > 0 and ready and time.time() - first_ready >= wait1 and time.time() - opened >= floor:
                    row["ended"] = f"{ready} group(s) ready for {wait1:.0f} s"
                    break
            pool2.active = False
            v["scheduler"].queue_groups = 1
            row["served"], row["moved_tries"] = pool2.served, pool2.used
            row["turn_waits"], row["turn_cap"] = getattr(pool2, "turn_waits", None), getattr(pool2, "turn_cap", None)
            row["moved_wait_s"] = round(route.waited, 1)
            route.close(getattr(v["card2"], "hold", False))
            row["sleep_s"] = await self.server2.sleep()
        except Exception as error:  # noqa: BLE001 - card 2 back to the trainer; a fresh server 2 or Mode B off
            if getattr(level_bench, "SHARED_SLOTS2", None) is not None:
                level_bench.SHARED_SLOTS2.active = False
            v["scheduler"].queue_groups = 1
            failures = v["s2_failures"] = int(v.get("s2_failures", 0)) + 1
            # timeshare.max_failures (v6h hang, v6i death: one failure turned card-2 serving off for the run): below
            # it, the dead server is stopped and the next block boots a fresh one (ts_boot, ~150-400 s)
            revive = failures < int((mg.get("timeshare") or {}).get("max_failures", 1))
            if getattr(self, "server2", None) is not None:
                row["server2_log"] = self.v7_keep_server2_log(failures)
                self.server2.stop()
            if v.get("route2") is not None:
                v["route2"].close(False)
                v["route2"].release()
            row["error"] = repr(error)[:300]
            if revive:
                self.server2 = None
                self.feed.emit("alert", why=f"v7 Mode B block failed ({failures}); server 2 boots fresh at the next "
                               "block", error=row["error"])
            else:
                self.ts_dead, v["card2"].on = True, False
                self.feed.emit("alert", why="v7 Mode B block failed; off for the rest of the run", error=row["error"])
        finally:
            row["unpark"] = await self.trainer.must("unpark")
        row["seconds"] = round(time.time() - began, 1)
        self.feed.emit("block_off", **row)
        return row
