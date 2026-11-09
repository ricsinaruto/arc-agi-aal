"""The RSI loop controller (docs/rsi.md, restart of 16 Oct): proposer sessions next to raced pool evaluations on one
GPU. Root process inside the bundle's level-bench trial (``arena.level_bench.run_level_bench``).

Fixed side (out of the proposer's reach): this controller, the gateway and scorer (``rsi2_loop.gateway``), the marks
and bars, the pool, the meter and the timer. The proposer submits candidate trees; each queued job (a pool evaluation
or a practice play) holds the GPU alone, the proposer's session runs next to it through the metering proxy.
"""
from __future__ import annotations

import asyncio
import json
import os
import shutil
import time
import traceback
from pathlib import Path

from . import candidates as cand
from .evaluate import Run, tree_problems
from .proposer import ProposerHost
from .proxy import Proxy
from .runtime import Runtime
from .serving import Stacks, clip, sample_gpu

DEFAULTS = {
    "root": "/rsi",
    "proxy_port": 18080,
    "slice_seconds": 30000 / 110,
    # marks as fractions of the pool's competition slice (len(pool) x slice_seconds): 50K / 150K / 380K tokens a game
    "mark_fractions": {"s1": 50 / 380, "s2": 150 / 380, "s3": 1.0},
    # 17 Oct noise check on the 29-game drop-rule pool: paired SE of two gen-0 runs 1.09 / 4.35 / 5.25;
    # the reference is one gen-0 run on the pool, so one run against one run: cut at -1 SE, accept at +2 SE
    # no s1 cutoff (s1 is still read and shown)
    "cut": {"s2": -4.4},
    "accept": 10.5,
    # once gen 0 has two pool runs, a candidate run is read against their mean: SE x sqrt(3)/2 (4.35 -> 3.77, 5.25 -> 4.55)
    "cut_two_refs": {"s2": -3.8},
    "accept_two_refs": 9.1,
    # (3a, 17 Oct) an s3 accept becomes final only after a confirmation: a second candidate run, read with its first as
    # a two-run mean against gen 0's two-run mean (SE 5.25 / sqrt(2) = 3.71; +2 SE)
    "confirm_accepts": False,
    "confirm_accept": 7.4,
    "proposer_continue": False,
    # continuous signal: per game, time to the 1st and 2nd level clear, censored at the mark
    # (evaluate.Run.scores_at "cont"). Noise runs on the 29 games, one run vs one run: paired SE 2.50 at s2, 2.25 at s3
    # (score 4.35 / 5.25); correlation of the paired score and cont differences 0.59 at s2, 0.31 at s3. Pre-registered
    # rule: z = (diff_score / SE_score + diff_cont / SE_cont) / sqrt(2 + 2r), SEs scaled for the number of runs on each
    # side; cut at s2 if z <= z_cut (0 from l6, futility; -1 before); accept at s3 if z >= 2 and both differences are positive; a confirmation (3a) uses the
    # same rule on the candidate's two-run mean.
    "continuous": False,
    "se": {"score": {"s1": 1.09, "s2": 4.35, "s3": 5.25}, "cont": {"s1": 2.09, "s2": 2.50, "s3": 2.25}},
    "r": {"s1": 0.27, "s2": 0.59, "s3": 0.31},
    "z_cut": 0.0,  # futility cut: was -1
    "z_accept": 2.0,
    # frontier-level evaluation: candidates first play the incumbent's
    # frontier levels from saved states, k levels x r attempts at once, against r_ref reference attempts per level;
    # a pass (combined z >= z, both differences > 0) goes on to the full-pool evaluation and its confirmation
    "frontier": False,
    "frontier_k": 8,
    "frontier_r": 6,
    "frontier_ref_r": 12,
    "frontier_sel_r": 2,
    "frontier_attempt_seconds": 91,
    "frontier_z_accept": 2.0,
    # split evaluation: every pool run plays half the pool here and half on
    # the worker VM ``split_peer`` (a run id), both at the per-game share (wall = games x slice); the rule reads the
    # merged scores. A split pool run is a different regime from a whole one, so gen 0 is measured split first
    # (``split_refs`` runs) and the whole-pool references are kept aside.
    "split_peer": None,
    "split_refs": 2,
    # ``split_mode``: "parallel": the worker VM plays whole pool runs of other
    # candidates at the same time, each VM against its own gen-0 reference (same regime, ``worker_refs`` gen-0 runs on
    # the worker); "halves": the half-pool split above.
    "split_mode": "parallel",
    "worker_refs": 1,
    "worker_failures_max": 3,
    "practice_n": 4,
    # practice batches: up to this many queued practice runs play
    # at once on the loop VM. A/B arms (same games and seconds) always play in the same batch, under the same load;
    # arms whose serving differs play one at a time, each alone. Pool evaluations start before queued practice.
    "practice_parallel": 3,
    "practice_games_max": 12,
    "practice_seconds_per_game_min": 60,
    # 18 Oct: a candidate stack that differs from gen 0's cannot start yet (the untrusted start fails its preflight);
    # its jobs wait in the queue instead of failing (coordinator)
    "hold_serving_changes": False,
    "budget_loop_gpu_hours": 100,
    "end_margin_seconds": 1200,
    "mirror_seconds": 600,
}


def atomic_write(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(data, indent=1, default=str))
    os.replace(tmp, path)


def games_share(job, batch):
    """A batch member's part of the eval memory ceiling: by its number of games (each game's sandbox needs about the
    same memory; l7w: two 12-game runs at a third each were killed at ~300 s)."""
    total = sum(max(1, len(j.get("games") or ())) for j in batch)
    return max(1, len(job.get("games") or ())) / total


def unpack_traces(data, target):
    """A worker's practice traces (tar.gz of plain files) into ``target``."""
    import io
    import tarfile

    target = Path(target)
    target.mkdir(parents=True, exist_ok=True)
    with tarfile.open(fileobj=io.BytesIO(data), mode="r:gz") as tar:
        members = [m for m in tar.getmembers() if m.isfile() and not m.name.startswith("/")
                   and ".." not in Path(m.name).parts]
        tar.extractall(target, members=members, filter="data")


class StackDown(RuntimeError):
    """The loop VM's serving stack is gone (a switch and both rollbacks failed)."""


class WorkerLost(Exception):
    """The worker VM failed, fell silent or lagged on a remote pool run; the job is played here instead."""

    def __init__(self, why, benign=False):
        super().__init__(why)
        self.benign = benign  # not the worker's fault (loop stop, a tree too big to send)


class Loop:
    def __init__(self, study, trial, *, source, games_dir, output, server_dir, base_url, deadline_epoch, log=print,
                 **_):
        self.cfg = {**DEFAULTS, **(study.get("rsi2_loop") or {})}
        self.study, self.source, self.output, self.log = study, Path(source), Path(output), log
        self.deadline = float(deadline_epoch)
        self.root = Path(self.cfg["root"])
        self.ckpt = self.output / "rsi2"
        self.ckpt.mkdir(parents=True, exist_ok=True)
        self.events = (self.ckpt / "events.jsonl").open("a")
        self.served = study.get("served_model") or self.served_from(self.source)
        self.upstream = base_url.rstrip("/")
        self.proxy = Proxy(self.upstream.removesuffix("/v1"), self.served, self.root / "root/proxy",
                           self.cfg["proxy_port"], unix_path=self.root / "psock/proxy.sock")
        self.runtime = Runtime(self.root, self.source, games_dir, self.cfg, self.proxy, self.ckpt)
        self.pool = list(self.cfg["pool"])
        full = len(self.pool) * self.cfg["slice_seconds"]
        self.cfg["marks"] = {m: round(full * f) for m, f in self.cfg["mark_fractions"].items()}
        self.stacks = Stacks(self.source, server_dir, self.upstream, self.served, self.root / "root/stacks",
                             log=self.event_log, runtime=self.runtime)
        self.state = self.load_state()
        self.dispatcher = None
        if int(self.cfg.get("practice_parallel", 1)) > 1:
            from . import proposer as P

            P.EXTRA_FACTS = (
                "- Practice runs share the GPU (from 17 Oct, leg l7): up to "
                f"{int(self.cfg['practice_parallel'])} queued practice runs play at once. Runs with the same games and "
                "seconds (the arms\n  of an A/B) always play together, under the same load, if they are all queued "
                "when the first starts and\n  their serving is the same; arms whose serving differs play one at a time, "
                "each alone. Each game therefore gets\n  fewer tokens within its wall than in practice runs before this "
                "change: do not compare practice results across it.\n  A run's result.json lists the runs it played "
                "with. Pool evaluations start before any queued practice run.\n")
        if self.cfg.get("loop_notes"):  # factual operator notices about the loop's own faults (never advice)
            from . import proposer as P

            P.EXTRA_FACTS += "".join(f"- Loop notice: {n}\n" for n in self.cfg["loop_notes"])
        if self.cfg["split_peer"]:
            from . import exchange as X

            base = Path(os.environ.get("ARC_VALIDATION_OUTPUT") or self.cfg.get("exchange_root") or self.output)
            self.dispatcher = X.Dispatcher(
                X.Mailbox(base / "rsi-exchange/out", base / "exchange-in" / self.cfg["split_peer"] / "rsi-exchange/out"),
                self.source / "rsi/gen0/tree")
            if self.cfg["split_mode"] == "parallel":
                from . import proposer as P

                P.EXTRA_FACTS = P.EXTRA_FACTS + ("- Two GPUs evaluate: a second machine (no proposer) plays queued pool evaluations too, "
                                 "each\n  machine against its own gen-0 runs; `rsi.status()` shows "
                                 "what runs there. When no pool evaluation waits, practice runs\n  play there too (all arms of an A/B on the same machine); result.json says which machine.\n")
            if self.cfg["split_mode"] == "halves" and self.state.get("regime") != "split":
                self.state["regime"] = "split"
                self.state["unsplit"] = {k: self.state.pop(k) for k in ("reference", "references", "ref_n", "se_loop")
                                         if k in self.state}
                self.state.pop("cont_ref_queued", None)
                self.state["split_refs_queued"] = False
        self.job_task = None
        self.current_job = None
        self.remote_task = None
        self.remote_job = None
        self.stopping = False
        self.started = time.time()

    @staticmethod
    def served_from(source):
        """The model name the server answers to (arena.run's rule: study served_model, else the agent config's)."""
        import yaml

        return yaml.safe_load((Path(source) / "configs/duck_controls.yaml").read_text())["model"]

    # ----- state -------------------------------------------------------------------------------------------------
    def load_state(self):
        for path in (self.ckpt / "state.json", self.source / "configs/rsi2_resume/state.json"):
            if path.is_file():
                state = json.loads(path.read_text())
                running = state.pop("running_job", None)
                remote = state.pop("running_remote_job", None)
                for job in (remote, running):
                    if job and job.get("kind") == "batch":
                        state["jobs"][:0] = job["jobs"]
                    elif job and not job.get("worker_ref"):
                        state["jobs"].insert(0, job)
                return state
        return {"seq": 0, "meters": {"loop_s": 0.0, "legs": 0}, "candidates": {}, "jobs": [], "runs": [],
                "practice": {}, "stack": {"live": "c0000", "live_hash": None, "history": []},
                "meta": {"live": None, "items": {}}, "proposer": {"sessions": []}, "accepted": []}

    def save(self, reason=""):
        state = dict(self.state)
        # the GPU-h meter as of now, so a leg that dies (or is deleted) keeps its hours on resume
        state["meters"] = {**self.state["meters"], "loop_s": self.state["meters"]["loop_s"] + time.time() - self.started}
        if self.job_task is not None and getattr(self, "current_job", None):
            state["running_job"] = self.current_job
        if getattr(self, "remote_task", None) is not None and getattr(self, "remote_job", None):
            state["running_remote_job"] = self.remote_job
        atomic_write(self.ckpt / "state.json", state)

    def event_log(self, row):
        row = {"t": time.time(), **row}
        self.events.write(json.dumps(row, default=str) + "\n")
        self.events.flush()
        self.log(json.dumps(row, default=str), flush=True)

    def new_id(self, prefix):
        self.state["seq"] += 1
        return f"{prefix}{self.state['seq']:04d}"

    def candidate_dir(self, cid):
        return self.root / "archive" / cid / "tree"

    def loop_hours(self):  # both VMs: the loop VM's wall plus the worker VM's while it is up (docs/rsi.md x-axis)
        return (self.state["meters"]["loop_s"] + self.state["meters"].get("worker_s", 0.0) + time.time()
                - self.started) / 3600

    def meter_worker(self):
        now = time.time()
        last, self._worker_tick = getattr(self, "_worker_tick", now), now
        if self.dispatcher is not None and not self.state.get("remote_disabled") and self.dispatcher.worker_alive():
            self.state["meters"]["worker_s"] = self.state["meters"].get("worker_s", 0.0) + min(now - last, 300)

    def stop_requested(self):
        local = os.environ.get("ARENA_STOP_REQUEST")  # the runner's copy of live/stop-request (arena.early_stop)
        return any(p.exists() for p in (self.output / "stop-request", self.output.parent / "stop-request",
                                        self.output.parent.parent / "live/stop-request",
                                        *([Path(local)] if local else [])))

    # ----- candidates and jobs ----------------------------------------------------------------------------------
    def register_candidate(self, cid, tree, parent, note, author, meta=None):
        problems = tree_problems(tree)
        item = {"parent": parent, "note": note[:4000], "author": author, "meta": meta, "created": time.time(),
                "serving_hash": cand.serving_hash(tree), "problems": problems, "marks": {},
                "status": "rejected" if problems else "queued", "verdict": None}
        self.state["candidates"][cid] = item
        atomic_write(self.root / "archive" / cid / "meta.json",
                     {k: item[k] for k in ("parent", "note", "author", "created", "problems")})
        if not problems:
            kind = "frontier" if self.cfg["frontier"] else "eval"
            self.state["jobs"].append({"id": self.new_id("j"), "kind": kind, "candidate": cid, "queued": time.time()})
        self.event_log({"event": "rsi2_candidate", "candidate": cid, "parent": parent, "status": item["status"],
                        "problems": problems[:5]})
        return item

    def queue_position(self, cid):
        ids = [j["candidate"] for j in self.state["jobs"]]
        return ids.index(cid) + 1 if cid in ids else None

    def withdraw(self, cid=None, job=None):
        """A queued job by id (practice or a candidate's evaluation), or a candidate's queued evaluation and practice."""
        if job:
            hit = [j for j in self.state["jobs"] if j["id"] == job]
            if not hit:
                return {"withdrawn": False, "why": f"{job} is not queued (running or done jobs cannot be withdrawn)"}
            if hit[0]["kind"] == "eval" and hit[0]["candidate"] == "c0000":
                return {"withdrawn": False, "why": "gen 0's reference run is fixed"}
            cid = hit[0]["candidate"] if hit[0]["kind"] == "eval" else None
            self.state["jobs"] = [j for j in self.state["jobs"] if j["id"] != job]
            if cid and cid in self.state["candidates"]:
                self.state["candidates"][cid]["status"] = "withdrawn"
            self.save("withdraw")
            return {"withdrawn": True, "job": job}
        if cid == "c0000":
            return {"withdrawn": False, "why": "gen 0's reference run is fixed; withdraw its practice by job id"}
        gone = [j["id"] for j in self.state["jobs"] if j["candidate"] == cid]
        evals = [j for j in self.state["jobs"] if j["candidate"] == cid and j["kind"] == "eval"]
        self.state["jobs"] = [j for j in self.state["jobs"] if j["candidate"] != cid]
        if evals and cid in self.state["candidates"]:
            self.state["candidates"][cid]["status"] = "withdrawn"
        self.save("withdraw")
        return {"withdrawn": bool(gone), "jobs": gone}

    def practice_games(self):
        if not hasattr(self, "_practice_ids"):
            ids = []
            for meta in sorted((self.root / "games/practice/environment_files").glob("*/*/metadata.json")):
                try:
                    ids.append(json.loads(meta.read_text())["game_id"])
                except (OSError, ValueError, KeyError):
                    continue
            self._practice_ids = sorted(set(ids))
        return self._practice_ids

    def request_practice(self, cid, games, n, seconds):
        if cid not in self.state["candidates"] or self.state["candidates"][cid]["status"] == "rejected":
            raise ValueError(f"{cid} is not an accepted candidate tree")
        known = self.practice_games()
        if games:
            unknown = [g for g in games if g not in known]
            if unknown:
                raise ValueError(f"not practice games: {unknown[:5]}")
        else:  # gen 0's default: the least-played practice games first
            plays = {g: len(self.state["practice"].get(g, [])) for g in known}
            games = sorted(known, key=lambda g: (plays[g], g))[:max(1, int(n))]
        games = list(games)[:self.cfg["practice_games_max"]]
        wall = float(seconds) if seconds else len(games) * self.cfg["slice_seconds"]
        if not 120 <= wall <= 4 * 3600:
            raise ValueError("seconds: 120..14400 (the play's wall for all its games at once)")
        floor = self.cfg["practice_seconds_per_game_min"]
        if wall / len(games) < floor:
            raise ValueError(f"seconds: at least {floor:g} per game ({len(games)} games -> {floor * len(games):g} s); "
                             f"a pool game gets {self.cfg['slice_seconds']:.1f} s")
        job = {"id": self.new_id("j"), "kind": "practice", "candidate": cid, "games": games, "seconds": wall,
               "queued": time.time()}
        self.state["jobs"].append(job)
        self.save("practice")
        return {"queued": job["id"], "games": games, "seconds": wall, "queue_position": len(self.state["jobs"])}

    def status_view(self):
        cands = self.state["candidates"]
        return {
            "budget": {"loop_gpu_hours": round(self.loop_hours(), 2),
                       "approved_loop_gpu_hours": self.cfg["budget_loop_gpu_hours"],
                       "this_leg_ends_in_hours": round((self.deadline - time.time()) / 3600, 2)},
            "running": getattr(self, "current_job", None),
            **({"running_on_worker_vm": getattr(self, "remote_job", None)} if self.dispatcher is not None else {}),
            "queue": [{k: j.get(k) for k in ("id", "kind", "candidate", "games", "seconds")} for j in self.state["jobs"]],
            "live_stack": self.state["stack"]["live"],
            **({"held_until_serving_fix": [j["id"] for j in self.state["jobs"] if self.held(j)]}
               if self.cfg.get("hold_serving_changes") else {}),
            "marks_seconds": self.cfg["marks"], "pool_games": self.pool,
            "candidates": {c: {"status": v["status"], "parent": v["parent"], "note": v["note"][:200],
                               "problems": v["problems"][:5], "verdict": v.get("verdict"),
                               "marks": {m: {"pool_mean": r.get("pool_mean"), "diff_vs_gen0": r.get("diff")}
                                         for m, r in v["marks"].items()}}
                           for c, v in cands.items()},
            "practice_games": self.practice_games(),
            "practice_rates": {g: [round(r["levels"] / max(1, r["total_levels"] or 1), 3) for r in rows]
                               for g, rows in self.state["practice"].items()},
            "meta_live": self.state["meta"]["live"],
            **({"frontier": {"incumbent": (self.state.get("frontier") or {}).get("incumbent"),
                             "phase": (self.state.get("frontier") or {}).get("phase"),
                             "levels": (self.state.get("frontier") or {}).get("selected"),
                             "reference": {k: {"attempts": len(v), "clear_rate": round(sum(x["y"] for x in v) / len(v), 3)}
                                           for k, v in ((self.state.get("frontier") or {}).get("ref") or {}).items()}}}
               if self.cfg["frontier"] else {}),
        }

    # ----- scoring ----------------------------------------------------------------------------------------------
    def refs(self, vm="loop"):
        """The dict holding a VM's gen-0 reference (``reference``, ``references``, ``ref_n``, ``se_loop``)."""
        return self.state if vm == "loop" else self.state.setdefault("vm_refs", {}).setdefault(vm, {})

    def bars(self, vm="loop"):
        two = len(self.refs(vm).get("references") or []) >= 2
        return (self.cfg["cut_two_refs"], self.cfg["accept_two_refs"]) if two else (self.cfg["cut"], self.cfg["accept"])

    def add_reference(self, games, vm="loop"):
        """One more gen-0 pool run ({game: {"s2": score, "s2_c": cont, ...}}): the reference is, key by key, the mean
        over the runs that have that key (l2's run has no cont). Each VM has its own (``vm``)."""
        box = self.refs(vm)
        refs = box.setdefault("references", [])
        if not refs and box.get("reference"):
            refs.append(box["reference"])  # a leg resumed from before the list existed
        refs.append(games)
        keys = sorted({k for r in refs for k in r[self.pool[0]]})
        box["ref_n"] = {k: sum(1 for r in refs if k in r[self.pool[0]]) for k in keys}
        box["reference"] = {g: {k: sum(r[g][k] for r in refs if k in r[g]) / box["ref_n"][k]
                                for k in keys} for g in self.pool}
        # loop-conditions noise check: the paired SE of the first two gen-0 loop runs, per key;
        # zstat uses the larger of it and the 85-game value (pre-registered: never a looser bar than measured)
        both = [r for r in refs[:2]]
        if len(both) == 2:
            se = {}
            for k in keys:
                if all(k in r[self.pool[0]] for r in both):
                    d = [both[0][g][k] - both[1][g][k] for g in self.pool]
                    mu = sum(d) / len(d)
                    se[k] = round((sum((x - mu) ** 2 for x in d) / (len(d) - 1)) ** .5 / len(d) ** .5, 3)
            box["se_loop"] = se
            self.event_log({"event": "rsi2_noise_check", "vm": vm, "paired_se": se})
        return len(refs)

    def ref_n(self, key, vm="loop"):
        box = self.refs(vm)
        return (box.get("ref_n") or {}).get(key, 1 if key in (box.get("reference") or {}).get(self.pool[0], {}) else 0)

    def zstat(self, mark, diff, diff_c, runs=1, vm="loop", ref_n=None):
        """The pre-registered combined z of a candidate (mean of ``runs`` runs) against gen 0's reference (``vm``'s).
        The score SE is the largest of the configured one and any VM's measured loop noise."""
        measured = [(b.get("se_loop") or {}).get(mark, 0) for b in [self.state, *(self.state.get("vm_refs") or {}).values()]]
        n_s = ref_n or max(1, self.ref_n(mark, vm))
        n_c = ref_n or max(1, self.ref_n(f"{mark}_c", vm))
        se_s = max(self.cfg["se"]["score"][mark], *measured) * ((1 / runs + 1 / n_s) / 2) ** .5
        se_c = self.cfg["se"]["cont"][mark] * ((1 / runs + 1 / n_c) / 2) ** .5
        return (diff / se_s + diff_c / se_c) / (2 + 2 * self.cfg["r"][mark]) ** .5

    def decide(self, cid, vm="loop"):
        ref = self.refs(vm).get("reference") or {}
        cut, accept = self.bars(vm)

        def verdict(mark, scores):
            item = self.state["candidates"][cid]
            if not scores:
                item["marks"][mark] = {"error": "no snapshot"}
                return "continue" if cid == "c0000" else "cut"
            games = scores["games"]
            pool = self.pool
            mean = sum(games.get(g, {}).get("score", 0.0) for g in pool) / len(pool)
            if cid == "c0000" and vm != "loop":  # a worker VM's gen-0 run: its own reference, not c0000's display
                self.refs(vm).setdefault("pending", {})[mark] = {g: games.get(g) for g in pool}
                self.event_log({"event": "rsi2_reference", "vm": vm, "mark": mark, "pool_mean": round(mean, 3)})
                return "reference"
            if cid == "c0000":  # the reference run: never cut
                item["marks"][mark] = {"pool_mean": round(mean, 3), "games": {g: games.get(g) for g in pool}}
                self.write_scores(cid)
                self.event_log({"event": "rsi2_reference", "mark": mark, "pool_mean": round(mean, 3)})
                return "reference"
            diffs = [games.get(g, {}).get("score", 0.0) - ref[g][mark] for g in pool]
            diff = sum(diffs) / len(diffs)
            item["marks"][mark] = {"pool_mean": round(mean, 3), "diff": round(diff, 3), "vm": vm,
                                   "games": {g: games.get(g) for g in pool}}
            cont_ok = self.cfg["continuous"] and f"{mark}_c" in ref[pool[0]]
            if cont_ok:
                cmean = sum(games.get(g, {}).get("cont", 0.0) for g in pool) / len(pool)
                diff_c = sum(games.get(g, {}).get("cont", 0.0) - ref[g][f"{mark}_c"] for g in pool) / len(pool)
                z = self.zstat(mark, diff, diff_c, vm=vm)
                item["marks"][mark].update(vm=vm, cont_mean=round(cmean, 3), diff_cont=round(diff_c, 3), z=round(z, 3))
                out = ("cut" if mark == "s2" and z <= self.cfg["z_cut"] else "continue" if mark != "s3" else
                       "accept" if z >= self.cfg["z_accept"] and diff > 0 and diff_c > 0 else "reject")
            elif mark in cut:
                out = "cut" if diff <= cut[mark] else "continue"
            elif mark != "s3":
                out = "continue"
            else:
                out = "accept" if diff >= accept else "reject"
            self.write_scores(cid)
            self.event_log({"event": "rsi2_verdict", "candidate": cid, "mark": mark, "diff": round(diff, 3),
                            "pool_mean": round(mean, 3), "verdict": out,
                            **{k: item["marks"][mark].get(k) for k in ("diff_cont", "z") if k in item["marks"][mark]}})
            return out
        return verdict

    def write_scores(self, cid):
        item = self.state["candidates"][cid]
        atomic_write(self.root / "archive" / cid / "scores.json",
                     {"status": item["status"], "verdict": item.get("verdict"),
                      **({"frontier": {k: item["frontier"][k] for k in ("levels", "z", "pass", "vs_incumbent")}}
                         if item.get("frontier") else {}),
                      "marks": {m: {"seconds": self.cfg["marks"].get(m), **{k: v for k, v in r.items() if k != "diff"}}
                                for m, r in item["marks"].items()}})
        self.runtime.seal(self.root / "archive" / cid)

    # ----- jobs ---------------------------------------------------------------------------------------------------
    async def ensure_stack(self, cid):
        """The candidate's serving/ is live (a switch if not); False when its stack failed (gen 0's runs again)."""
        item = self.state["candidates"][cid]
        stack = self.state["stack"]
        if item["serving_hash"] == stack["live_hash"]:
            return True
        self.proxy.open.clear()
        try:
            held = time.time()
            while sum(self.proxy.inflight.values()) > 0 and time.time() - held < 900:
                await asyncio.sleep(2)
            old = stack["live"]
            try:
                record = await self.stacks.switch(self.candidate_dir(cid) / "serving", cid,
                                                  self.candidate_dir(old) / "serving", old,
                                                  fallback=(self.candidate_dir("c0000") / "serving", "c0000"))
            except Exception as exc:  # nothing came back (gen 0's own start failed too): infrastructure, not the
                why = clip(f"{type(exc).__name__}: {exc}")  # candidate; the leg stops and the job waits for the next
                self.event_log({"event": "rsi2_stack_down", "candidate": cid, "error": why})
                (self.output / "stop-request").write_text(f"serving stack down at {cid}: {why}\n")
                raise StackDown(why) from exc
            live = {"new": cid, "old": old, "fallback": "c0000"}[record["live"]]
            stack.update(live=live, live_hash=self.state["candidates"][live]["serving_hash"])
            stack["history"].append({k: record.get(k) for k in ("from", "to", "ok", "live", "seconds", "error")})
            self.event_log({"event": "rsi2_stack", **{k: record.get(k) for k in ("from", "to", "ok", "live",
                                                                                   "seconds", "error")}})
            if not record.get("ok"):
                item["problems"].append(f"serving: {record.get('error')}"[:600])
            return bool(record.get("ok"))
        finally:
            self.proxy.open.set()

    async def run_job(self, job, vm="loop"):
        cid = job["candidate"]
        item = self.state["candidates"][cid]
        try:
            if vm == "loop" and not await self.ensure_stack(cid):
                item["status"] = "rejected"
                self.notify({"event": "candidate_rejected", "candidate": cid, "why": item["problems"][-1:]})
                return
            if job["kind"] == "eval" and cid == "c0000" and vm != "loop":  # the worker VM's own gen-0 reference
                self.refs(vm)["pending"] = {}
                result, run = await self.pool_run(cid, job, self.decide(cid, vm), list(self.cfg["marks"]), vm=vm)
                pending = self.refs(vm).pop("pending", {})
                if set(pending) != set(self.cfg["marks"]):
                    raise RuntimeError(f"{vm} reference run incomplete ({result['status']}, marks {sorted(pending)})")
                run_games = {g: {} for g in self.pool}
                for m in self.cfg["marks"]:
                    for g in self.pool:
                        row = pending[m].get(g) or {}
                        run_games[g][m] = float(row.get("score") or 0.0)
                        if "cont" in row:
                            run_games[g][f"{m}_c"] = float(row["cont"])
                n = self.add_reference(run_games, vm=vm)
                self.state["runs"].append({k: v for k, v in result.items() if k != "marks"})
                self.event_log({"event": "rsi2_vm_reference", "vm": vm, "runs": n, "pool_mean": {
                    m: round(sum(r[m] for r in run_games.values()) / len(self.pool), 3) for m in self.cfg["marks"]}})
                return
            if job["kind"] == "eval":
                item["status"] = "evaluating" if vm == "loop" else f"evaluating on the {vm} VM"
                item["marks"] = {}
                result, run = await self.pool_run(cid, job, self.decide(cid, vm), list(self.cfg["marks"]), vm=vm)
                item["vm"] = vm
                item["pool_logs"] = item.get("pool_logs", []) + ([str(run.actions)] if vm == "loop" else [])
                verdicts = {m: r.get("verdict") for m, r in result["marks"].items()}
                if cid == "c0000":
                    if set(verdicts) != set(self.cfg["marks"]):
                        raise RuntimeError(f"reference run incomplete ({result['status']}, marks {sorted(verdicts)}): "
                                           + (result.get("log_tail") or "")[-1500:])
                    run_games = {}
                    for g in self.pool:
                        run_games[g] = {}
                        for m in self.cfg["marks"]:
                            row = item["marks"][m]["games"].get(g) or {}
                            run_games[g][m] = float(row.get("score") or 0.0)
                            if "cont" in row:
                                run_games[g][f"{m}_c"] = float(row["cont"])
                    n = self.add_reference(run_games)
                    ref = self.state["reference"]
                    item["marks"] = {m: {"pool_mean": round(sum(ref[g][m] for g in self.pool) / len(self.pool), 3),
                                         **({"cont_mean": round(sum(ref[g][f"{m}_c"] for g in self.pool) / len(self.pool), 3)}
                                            if f"{m}_c" in ref[self.pool[0]] else {}),
                                         "games": {g: {"score": round(ref[g][m], 3),
                                                       **({"cont": round(ref[g][f"{m}_c"], 3)} if f"{m}_c" in ref[g]
                                                          else {})} for g in self.pool}}
                                     for m in self.cfg["marks"]}
                    item.update(status="reference", verdict=f"reference (mean of {n} gen-0 run{'s' * (n > 1)} on the pool)")
                    self.write_scores(cid)
                    self.notify({"event": "reference_done", "marks": {m: {"pool_mean": r.get("pool_mean")}
                                                                       for m, r in item["marks"].items()}})
                    self.state["runs"].append({k: v for k, v in result.items() if k != "marks"})
                    fr = self.state.get("frontier") or {}
                    if self.cfg["frontier"] and fr.get("incumbent", "c0000") == "c0000" and fr.get("phase") in (
                            None, "need_log"):
                        await self.frontier_levels("c0000", run)
                    return
                item["verdict"] = verdicts.get("s3") or ("cut at " + result["cut"] if result.get("cut") else
                                                         f"incomplete ({result['status']})")
                item["status"] = "accepted" if verdicts.get("s3") == "accept" else "done"
                if item["status"] == "accepted" and self.cfg["confirm_accepts"]:
                    item.update(status="confirming", verdict="accept at s3, confirmation queued")
                    self.state["jobs"].insert(0, {"id": self.new_id("j"), "kind": "confirm", "candidate": cid,
                                                  "queued": time.time()})
                    if len(self.state.get("references") or [self.state.get("reference")]) < 2:
                        self.state["jobs"].insert(0, {"id": self.new_id("j"), "kind": "eval", "candidate": "c0000",
                                                      "queued": time.time()})
                if item["status"] == "accepted":
                    self.state["accepted"].append({"candidate": cid, "at": time.time(),
                                                   "diff_s3": item["marks"]["s3"].get("diff")})
                self.write_scores(cid)
                self.notify({"event": "evaluation_done", "candidate": cid, "verdict": item["verdict"],
                             "marks": {m: {"pool_mean": r.get("pool_mean")} for m, r in item["marks"].items()}})
            elif job["kind"] in ("frontier", "fsel", "fref"):
                await self.run_frontier_job(job, item)
            elif job["kind"] == "confirm":
                s3 = self.cfg["marks"]["s3"]
                seen = {}
                result, run = await self.pool_run(cid, job, lambda mark, scores: seen.update(scores=scores) or
                                                  "continue", ["s3"], vm=vm)
                games = (seen.get("scores") or {}).get("games")
                if not games:
                    raise RuntimeError(f"confirmation run incomplete ({result['status']})")
                # each run against its own VM's gen-0 reference
                vm1 = item.get("vm", "loop")
                ref1, ref = self.refs(vm1)["reference"], self.refs(vm)["reference"]
                first = item["marks"]["s3"]["games"]
                pair = [((first.get(g) or {}).get("score", 0.0) - ref1[g]["s3"] + games.get(g, {}).get("score", 0.0)
                         - ref[g]["s3"]) / 2 for g in self.pool]
                diff = sum(pair) / len(self.pool)
                ok = diff >= self.cfg["confirm_accept"]
                if self.cfg["continuous"] and "s3_c" in ref[self.pool[0]] and "s3_c" in ref1[self.pool[0]]:
                    pair_c = [((first.get(g) or {}).get("cont", 0.0) - ref1[g]["s3_c"] + games.get(g, {}).get(
                        "cont", 0.0) - ref[g]["s3_c"]) / 2 for g in self.pool]
                    diff_c = sum(pair_c) / len(self.pool)
                    n_ref = min(max(1, self.ref_n("s3", vm1)), max(1, self.ref_n("s3", vm)))
                    z = self.zstat("s3", diff, diff_c, runs=2, ref_n=n_ref)
                    ok = z >= self.cfg["z_accept"] and diff > 0 and diff_c > 0
                    item.setdefault("confirmation", {})
                    item["confirmation_z"] = {"diff_cont_two_runs": round(diff_c, 3), "z": round(z, 3)}
                item["confirmation"] = {"diff_two_runs": round(diff, 3), "pool_mean_run2": round(
                    sum(games.get(g, {}).get("score", 0.0) for g in self.pool) / len(self.pool), 3),
                    "games_run2": {g: games.get(g) for g in self.pool}, "confirmed": ok}
                item.update(status="accepted" if ok else "done",
                            verdict="accept, confirmed" if ok else "accept at s3, not confirmed")
                if ok:
                    self.state["accepted"].append({"candidate": cid, "at": time.time(), "diff_two_runs": round(diff, 3)})
                    if self.cfg["frontier"]:  # the confirmed candidate is the new incumbent
                        await self.switch_incumbent(cid, games)
                self.event_log({"event": "rsi2_confirmation", "candidate": cid, "diff": round(diff, 3), "confirmed": ok})
                self.write_scores(cid)
                self.notify({"event": "confirmation_done", "candidate": cid, "verdict": item["verdict"],
                             "diff_two_runs": round(diff, 3)})
            else:
                result = await self.play_practice(job)
            self.state["runs"].append({k: v for k, v in result.items() if k != "marks"})
        except StackDown:
            if cid != "c0000":
                item.update(status="queued", marks={})
            self.state["jobs"].insert(0, {k: v for k, v in job.items() if k != "local_only"})
        except (WorkerLost, asyncio.CancelledError) as exc:  # the job goes back to the queue, to run here
            if vm == "loop":
                raise
            self.state["remote_failures"] = self.state.get("remote_failures", 0) + (isinstance(exc, WorkerLost)
                                                                                  and not exc.benign)
            self.event_log({"event": "rsi2_worker_lost", "job": job["id"], "candidate": cid, "why": str(exc)[:300],
                            "failures": self.state["remote_failures"]})
            if not job.get("worker_ref"):
                if cid != "c0000":
                    item.update(status="queued", marks={})
                self.state["jobs"].insert(0, {**job, "local_only": True})
            if isinstance(exc, asyncio.CancelledError):
                raise
        except Exception as exc:  # noqa: BLE001 - the loop goes on
            self.event_log({"event": "rsi2_job_error", "job": job["id"], "vm": vm,
                            "error": clip(f"{type(exc).__name__}: {exc}"), "traceback": traceback.format_exc()[-2000:]})
            if job["kind"] == "eval" and cid == "c0000":
                item["status"] = "reference-pending"
                self.state.setdefault("reference_failures", 0)
                self.state["reference_failures"] += 1
            elif job["kind"] == "eval":
                item["status"] = "error"
                item["problems"].append(f"evaluation error: {type(exc).__name__}: {exc}"[:600])
        finally:
            if vm == "loop":
                self.current_job = None
            else:
                self.remote_job = None
            self.save("job")

    # ----- pool runs (whole, or split across the worker VM) ------------------------------------------------------
    def pool_wall(self):
        if self.dispatcher is None or self.cfg["split_mode"] != "halves":
            return max(self.cfg["marks"].values())
        from . import exchange as X

        a, b = X.split_games(self.pool)
        return max(len(a), len(b)) * self.cfg["slice_seconds"] + 1200  # + the exchange's latency and a late worker

    async def pool_run(self, cid, job, decide, names, vm="loop"):
        """(result, local Run): result["marks"][m] = {"verdict", "games"}; ``decide(mark, scores)`` sees the whole
        pool's per-game scores at each mark in ``names``."""
        if vm != "loop":
            return await self.remote_pool_run(cid, job, decide, names)
        if self.dispatcher is None or self.cfg["split_mode"] != "halves":
            run = Run(self, cid, self.candidate_dir(cid), kind="eval", games=self.pool,
                      games_dir=self.root / "games/pool/environment_files", wall=max(self.cfg["marks"][m] for m in names),
                      marks={m: self.cfg["marks"][m] for m in names}, tag=f"{cid}-{job['id']}", log_actions=True)
            return await run.go(decide), run
        return await self.split_pool_run(cid, job, decide, names)

    async def play_practice(self, job, share=1.0, batch_ids=()):
        """One practice run (archive/<cid>/practice/<job>/); ``share``: its part of the eval memory in a batch."""
        cid = job["candidate"]
        run = Run(self, cid, self.candidate_dir(cid), kind="practice", games=job["games"],
                  games_dir=self.root / "games/practice/environment_files", wall=job["seconds"],
                  tag=f"{cid}-{job['id']}")
        run.mem_share = share
        result = await run.go()
        final = run.scores_at(job["seconds"] + 3600, result.get("started") or time.time())
        target = self.root / "archive" / cid / "practice" / job["id"]
        target.mkdir(parents=True, exist_ok=True)
        games_out = run.out / "duck-transfer/games"
        if games_out.is_dir():
            await asyncio.to_thread(shutil.copytree, games_out, target / "games", symlinks=False,
                                    dirs_exist_ok=True,
                                    ignore=shutil.ignore_patterns("*.sqlite*", "*.db"))
        self.practice_done(job, (final or {}).get("games"), result["status"], batch_ids)
        return result

    def practice_done(self, job, games, status, batch_ids=(), vm="loop"):
        """Record a finished practice run (its traces are already under archive/<cid>/practice/<job>/games)."""
        cid = job["candidate"]
        target = self.root / "archive" / cid / "practice" / job["id"]
        target.mkdir(parents=True, exist_ok=True)
        atomic_write(target / "result.json", {"games": job["games"], "seconds": job["seconds"], "scores": games,
                                              "status": status, "machine": vm,
                                              "played_at_once_with": [b for b in batch_ids if b != job["id"]]})
        for game, row in (games or {}).items():
            self.state["practice"].setdefault(game, []).append(
                {"levels": row.get("levels") or 0, "total_levels": None, "candidate": cid})
        self.runtime.seal(self.root / "archive" / cid)
        self.notify({"event": "practice_done", "candidate": cid, "job": job["id"],
                     "traces": f"/workspace/archive/{cid}/practice/{job['id']}/"})

    async def run_batch(self, batch):
        """Practice runs played at once on the live stack (one serving hash), each like run_job's practice."""
        try:
            if not await self.ensure_stack(batch[0]["candidate"]):
                for job in batch:
                    self.state["candidates"][job["candidate"]]["status"] = "rejected"
                return
            async def one(job):
                try:
                    result = await self.play_practice(job, games_share(job, batch), [j["id"] for j in batch])
                    self.state["runs"].append({k: v for k, v in result.items() if k != "marks"})
                except Exception as exc:  # noqa: BLE001 - the other runs go on
                    self.event_log({"event": "rsi2_job_error", "job": job["id"], "vm": "loop",
                                    "error": clip(f"{type(exc).__name__}: {exc}"),
                                    "traceback": traceback.format_exc()[-2000:]})
            await asyncio.gather(*(one(job) for job in batch))
        finally:
            self.current_job = None
            self.save("job")

    def held(self, job):
        """A job of a candidate whose serving differs from gen 0's waits while serving changes cannot start."""
        if not self.cfg.get("hold_serving_changes") or job["candidate"] == "c0000":
            return False
        cands = self.state["candidates"]
        return (cands.get(job["candidate"]) or {}).get("serving_hash") != (cands.get("c0000") or {}).get("serving_hash")

    def local_index(self):
        """The next job for the loop VM: pool evaluations (and frontier jobs) before practice, else the head; None
        when every queued job is held."""
        free = [i for i, job in enumerate(self.state["jobs"]) if not self.held(job)]
        return next((i for i in free if self.state["jobs"][i]["kind"] != "practice"), free[0] if free else None)

    def practice_batch(self, i):
        """Queue indices of the practice runs that play together with practice job ``i``."""
        jobs, cap = self.state["jobs"], max(1, int(self.cfg.get("practice_parallel", 1)))
        sig = lambda j: (tuple(j.get("games") or ()), round(float(j.get("seconds") or 0)))
        serving = lambda j: (self.state["candidates"].get(j["candidate"]) or {}).get("serving_hash")
        groups = {}
        for k, job in enumerate(jobs):
            if job["kind"] == "practice":
                groups.setdefault(sig(job), []).append(k)
        head = groups[sig(jobs[i])]
        if cap == 1 or len({serving(jobs[k]) for k in head}) > 1:
            return [i]  # arms with different serving: one at a time, each alone (the same load: none)
        chosen = head[:cap]
        for key, ks in groups.items():
            if key == sig(jobs[i]) or len(chosen) + len(ks) > cap:
                continue
            if {serving(jobs[k]) for k in ks} == {serving(jobs[i])}:
                chosen += ks
        return sorted(chosen)

    async def run_remote_practice(self, batch):
        """A practice batch on the worker VM: posted together (they start together there), traces come back as
        results/<jid>/practice.tar.gz and land in the archive as if played here. On a worker failure the unfinished
        runs go back to the queue, to play here."""
        from . import exchange as X

        disp = self.dispatcher
        bid = self.new_id("b")
        jids = {f"{j['candidate']}-{j['id']}-w": j for j in batch}
        done = set()
        try:
            for jid, job in jids.items():
                sent = await asyncio.to_thread(
                    disp.post, jid, job["candidate"], self.candidate_dir(job["candidate"]), job["games"],
                    job["seconds"], {}, {"kind": "practice", "batch": bid, "batch_n": len(batch),
                                         "share": games_share(job, batch)})
                if not sent:
                    raise WorkerLost("tree too big to send", benign=True)
            posted = disp.now()
            final_seen = {}
            while len(done) < len(jids):
                for jid, job in jids.items():
                    if jid in done:
                        continue
                    final = disp.final(jid)
                    if final is None:
                        if disp.lagging(jid, "end", float(job["seconds"]) + 600, posted):
                            raise WorkerLost(f"worker late on practice {job['id']}")
                        continue
                    info = final.get("practice")
                    if not info:
                        raise WorkerLost(f"practice {job['id']} ended without traces: {final.get('status')}")
                    final_seen.setdefault(jid, disp.now())
                    try:
                        data = disp.box.peer_path(f"results/{jid}/practice.tar.gz").read_bytes()
                    except OSError:
                        data = None
                    if data is None or X.sha_bytes(data) != info.get("sha"):
                        if disp.now() - final_seen[jid] > X.LATE:
                            raise WorkerLost(f"practice {job['id']}: traces never arrived")
                        continue
                    target = self.root / "archive" / job["candidate"] / "practice" / job["id"] / "games"
                    await asyncio.to_thread(unpack_traces, data, target)
                    self.practice_done(job, info.get("scores"), final.get("status"), [j["id"] for j in batch],
                                       vm="worker")
                    self.state["runs"].append({"id": jid, "candidate": job["candidate"], "kind": "practice",
                                               "vm": "worker", "status": final.get("status"),
                                               "started": final.get("started"), "ended": final.get("ended")})
                    done.add(jid)
                    self.save("practice")
                if len(done) == len(jids):
                    break
                if not disp.worker_alive():
                    raise WorkerLost("worker silent")
                if self.stop_requested():
                    raise WorkerLost("loop stop", benign=True)
                await asyncio.sleep(self.cfg.get("split_poll", 15))
            self.state["remote_failures"] = 0
        except (WorkerLost, asyncio.CancelledError) as exc:
            for jid in jids:
                if jid not in done:
                    disp.cancel(jid, "taken back by the loop")
            back = [{**j, "local_only": True} for jid, j in jids.items() if jid not in done]
            self.state["jobs"][:0] = back
            benign = isinstance(exc, asyncio.CancelledError) or exc.benign
            self.state["remote_failures"] = self.state.get("remote_failures", 0) + (not benign)
            self.event_log({"event": "rsi2_worker_lost", "job": "+".join(j["id"] for j in batch),
                            "candidate": batch[0]["candidate"], "why": str(exc)[:300] or type(exc).__name__,
                            "failures": self.state["remote_failures"], "requeued": [j["id"] for j in back]})
            if isinstance(exc, asyncio.CancelledError):
                raise
        finally:
            self.remote_job = None
            self.save("job")

    async def remote_pool_run(self, cid, job, decide, names):
        """A whole pool run on the worker VM (split_mode "parallel"): the worker plays, the marks come back through
        the exchange and the rule reads them here. Raises WorkerLost when the worker fails, falls silent or lags."""
        disp = self.dispatcher
        marks = {m: self.cfg["marks"][m] for m in names}
        wall = max(marks.values())
        jid = f"{cid}-{job['id']}-w"
        tree = self.candidate_dir(cid)
        sent = await asyncio.to_thread(disp.post, jid, cid, tree, self.pool, wall, marks)
        if not sent:
            raise WorkerLost("tree too big to send", benign=True)
        posted = disp.now()
        self.event_log({"event": "rsi2_remote_post", "jid": jid, "candidate": cid, "wall": wall})
        out = {"marks": {}, "cut": None, "vm": "worker", "jid": jid}
        try:
            for m in sorted(names, key=lambda k: marks[k]):
                while True:
                    scores = disp.mark(jid, m)
                    if scores is not None:
                        break
                    final = disp.final(jid)
                    if final is not None:
                        raise WorkerLost(f"worker ended without {m}: {final.get('status')}")
                    if not disp.worker_alive():
                        raise WorkerLost("worker silent")
                    if disp.lagging(jid, m, marks[m], posted):
                        raise WorkerLost(f"worker late at {m}")
                    if self.stop_requested():
                        raise WorkerLost("loop stop", benign=True)
                    await asyncio.sleep(self.cfg.get("split_poll", 15))
                verdict = decide(m, scores if scores.get("games") else None)
                out["marks"][m] = {"verdict": verdict, "seconds": marks[m]}
                self.event_log({"event": "rsi2_mark", "run": jid, "candidate": cid, "mark": m, "verdict": verdict,
                                "vm": "worker"})
                self.save("mark")
                if verdict == "cut":
                    out["cut"] = m
                    break
        except BaseException:
            disp.cancel(jid, "taken back by the loop")
            raise
        if out["cut"]:
            disp.cancel(jid, f"cut at {out['cut']}")
        self.state["remote_failures"] = 0
        final = disp.final(jid) or {}
        out.update(status="complete" if out["cut"] or final.get("status") in (None, "complete") else final["status"],
                   started=final.get("started"), ended=time.time(), seconds=wall, exit=0)

        class _Remote:  # no local files for a remote run
            actions = Path("/nonexistent")
        return out, _Remote

    def launch_remote(self):
        """split_mode "parallel": keep the worker VM busy with whole pool runs; jobs still start in queue order."""
        if self.dispatcher is None or self.cfg["split_mode"] != "parallel" or self.stopping:
            return
        if self.remote_task is not None:
            if not self.remote_task.done():
                return
            self.remote_task = None
        if self.state.get("remote_failures", 0) >= self.cfg["worker_failures_max"]:
            if not self.state.get("remote_disabled"):
                self.state["remote_disabled"] = True
                self.event_log({"event": "rsi2_worker_disabled", "failures": self.state["remote_failures"]})
            return
        if not self.dispatcher.worker_alive():
            return
        box = self.refs("worker")
        if len(box.get("references") or []) < self.cfg["worker_refs"]:
            job = {"id": self.new_id("j"), "kind": "eval", "candidate": "c0000", "queued": time.time(),
                   "worker_ref": True}
        else:
            if not self.state.get("reference"):
                return
            # pool evaluations start before practice (which, with frontier jobs and the loop VM's own references,
            # stays on the loop VM): the first such evaluation in the queue
            pick = next((k for k, j in enumerate(self.state["jobs"]) if j["kind"] in ("eval", "confirm")
                         and j["candidate"] != "c0000" and not j.get("local_only") and not self.held(j)), None)
            if pick is None:  # no pool evaluation waiting: the next practice batch (all its A/B arms) plays there
                first = next((k for k, j in enumerate(self.state["jobs"]) if j["kind"] == "practice"
                              and not j.get("local_only") and not self.held(j)), None)
                if first is None or not self.cfg.get("worker_practice", True):
                    return
                picks = [k for k in self.practice_batch(first) if not self.state["jobs"][k].get("local_only")]
                batch = [self.state["jobs"][k] for k in picks]
                if time.time() + max(float(j["seconds"]) for j in batch) + 2400 > self.deadline:
                    return
                for k in reversed(picks):
                    self.state["jobs"].pop(k)
                self.remote_job = {"kind": "batch", "id": "+".join(j["id"] for j in batch), "jobs": batch}
                for j in batch:
                    self.event_log({"event": "rsi2_job_start", "vm": "worker", "batch": len(batch),
                                    **{k: j.get(k) for k in ("id", "kind", "candidate")}})
                self.remote_task = asyncio.create_task(self.run_remote_practice(batch))
                return
            job = self.state["jobs"].pop(pick)
        if time.time() + max(self.cfg["marks"].values()) + 2400 > self.deadline:
            if not job.get("worker_ref"):
                self.state["jobs"].insert(0, job)
            return
        self.remote_job = job
        self.event_log({"event": "rsi2_job_start", "vm": "worker",
                        **{k: job.get(k) for k in ("id", "kind", "candidate")}})
        self.remote_task = asyncio.create_task(self.run_job(job, vm="worker"))

    async def split_pool_run(self, cid, job, decide, names):
        from . import exchange as X

        disp = self.dispatcher
        a, b = X.split_games(self.pool)
        fractions = {m: self.cfg["mark_fractions"][m] for m in names}
        wall_a, marks_a = X.half_plan(a, self.cfg["slice_seconds"], fractions)
        wall_b, marks_b = X.half_plan(b, self.cfg["slice_seconds"], fractions)
        jid = f"{cid}-{job['id']}-b"
        tree = self.candidate_dir(cid)
        sent = await asyncio.to_thread(disp.post, jid, cid, tree, b, wall_b, marks_b)
        posted = disp.now()
        self.event_log({"event": "rsi2_split_post", "jid": jid, "games_here": len(a), "games_worker": len(b),
                        "wall": [wall_a, wall_b], "sent": sent})
        local, rerun = {}, {}
        run_a = Run(self, cid, tree, kind="eval", games=a, games_dir=self.root / "games/pool/environment_files",
                    wall=wall_a, marks=marks_a, tag=f"{cid}-{job['id']}-a", log_actions=True)
        task_a = asyncio.create_task(run_a.go(lambda m, sc: local.__setitem__(m, sc) or "continue"))
        out = {"marks": {}, "cut": None, "split": {"here": a, "worker": b, "jid": jid, "worker_mode": "worker"}}
        if not sent:  # tree delta too big for the interim sync: both halves here, one after the other
            out["split"].update(worker_mode="local", worker_failure="tree too big to send")
        run_b = task_b = None
        pending = sorted(names, key=lambda m: fractions[m])
        while pending:
            m = pending[0]
            if out["split"]["worker_mode"] == "worker":
                final = disp.final(jid)
                failed = bool(final) and m not in (final.get("marks") or {})
                if disp.mark(jid, m) is None and (failed or not disp.worker_alive()
                                                  or disp.lagging(jid, m, marks_b[m], posted)):
                    out["split"]["worker_mode"] = "local"
                    out["split"]["worker_failure"] = (final or {}).get("status") or "silent or late"
                    disp.cancel(jid, "taken back by the loop")
                    self.event_log({"event": "rsi2_split_takeback", "jid": jid, "mark": m,
                                    "why": out["split"]["worker_failure"]})
            if out["split"]["worker_mode"] == "local" and task_b is None and task_a.done():
                run_b = Run(self, cid, tree, kind="eval", games=b, games_dir=self.root / "games/pool/environment_files",
                            wall=wall_b, marks=marks_b, tag=f"{cid}-{job['id']}-b-here")
                task_b = asyncio.create_task(run_b.go(lambda mk, sc: rerun.__setitem__(mk, sc) or "continue"))
            remote = disp.mark(jid, m) if out["split"]["worker_mode"] == "worker" else rerun.get(m)
            remote_over = out["split"]["worker_mode"] == "local" and task_b is not None and task_b.done()
            here = local.get(m)
            if (here is not None or task_a.done()) and (remote is not None or remote_over):
                merged = X.merge_marks(here, remote) if (here or remote) else None
                verdict = decide(m, merged if merged and merged["games"] else None)
                out["marks"][m] = {"seconds": {"here": marks_a[m], "worker": marks_b[m]}, "verdict": verdict,
                                   "games": (merged or {}).get("games")}
                self.event_log({"event": "rsi2_mark", "run": jid[:-2], "candidate": cid, "mark": m, "verdict": verdict,
                                "split": out["split"]["worker_mode"]})
                self.save("mark")
                pending.pop(0)
                if verdict == "cut":
                    out["cut"] = m
                    run_a.stop_request(f"cut at {m}")
                    if run_b is not None:
                        run_b.stop_request(f"cut at {m}")
                    if out["split"]["worker_mode"] == "worker":
                        disp.cancel(jid, f"cut at {m}")
                    break
                continue
            if self.stop_requested():
                run_a.stop_request("loop stop")
                if run_b is not None:
                    run_b.stop_request("loop stop")
                disp.cancel(jid, "loop stop")
            if time.time() - getattr(self, "_last_beat", 0) > X.HEARTBEAT:
                self._last_beat = time.time()
                await asyncio.to_thread(disp.beat)  # heartbeat + re-sends while the halves play
            await asyncio.sleep(self.cfg.get("split_poll", 15))
        res_a = await task_a
        res_b = await task_b if task_b is not None else (disp.final(jid) or {"status": "running on the worker"})
        out.update(started=res_a.get("started"), ended=time.time(), seconds=res_a.get("seconds"),
                   status="complete" if res_a.get("status") == "complete" and res_b.get("status") in (
                       "complete", "running on the worker") else f"{res_a.get('status')} / {res_b.get('status')}",
                   exit=res_a.get("exit"))
        return out, run_a

    # ----- frontier levels ----------------------------------------------------------------------------------------
    def frontier_seconds(self, job):
        return len(self.frontier_instances(job)) * self.cfg["frontier_attempt_seconds"]

    def frontier_instances(self, job):
        """The attempt list of a frontier job: [spec key, ...] (each key repeated per attempt)."""
        fr = self.state["frontier"]
        if job["kind"] == "fsel":
            keys, n = list(fr.get("specs") or {}), self.cfg["frontier_sel_r"]
        elif job["kind"] == "fref":
            keys, n = fr.get("selected") or [], self.cfg["frontier_ref_r"]
        else:
            keys, n = fr.get("selected") or [], self.cfg["frontier_r"]
        return [k for _ in range(n) for k in keys]  # interleaved: attempt i of every level before attempt i + 1

    async def frontier_levels(self, incumbent, run):
        """From the incumbent's logged pool run: candidate levels, CPU-verified replays, then the selection batch."""
        from . import frontier as F

        fr = self.state["frontier"]
        log = F.read_log(run.actions)
        snaps = run.snapshots.read_text().splitlines() if run.snapshots.exists() else []
        last = json.loads(snaps[-1]) if snaps else {"source": {}, "cards": {}}
        card = max(last["cards"].values(), key=lambda c: len(c["games"])) if last["cards"] else {"games": {}}
        totals = {last["source"].get(e, e): (row.get("level_count") or 10 ** 6) for e, row in card["games"].items()}
        specs = F.candidate_levels(log, last["source"], totals)
        kept, dropped = await asyncio.to_thread(F.verify, self.root / "games/pool/environment_files", specs)
        fr.update(incumbent=incumbent, phase="need_selection", dropped=[
            {k: d[k] for k in ("source", "level", "dropped")} for d in dropped],
                  specs={f"{k['source']}@{k['level']}": k for k in kept}, selected=[], ref={})
        self.event_log({"event": "rsi2_frontier_levels", "incumbent": incumbent, "candidates": len(specs),
                        "kept": len(kept), "dropped": fr["dropped"][:10]})
        if kept:
            self.state["jobs"].insert(0, {"id": self.new_id("j"), "kind": "fsel", "candidate": incumbent,
                                          "queued": time.time()})
        else:
            fr["phase"] = "failed"  # candidates fall back to the full-pool evaluation

    async def switch_incumbent(self, cid, run2_games):
        """A confirmed candidate becomes the incumbent: its two pool runs are the pool reference, its first (logged)
        pool run gives the next frontier levels."""
        item = self.state["candidates"][cid]
        first = {g: {} for g in self.pool}
        for m in self.cfg["marks"]:
            for g in self.pool:
                row = (item["marks"].get(m) or {}).get("games", {}).get(g) or {}
                first[g][m] = float(row.get("score") or 0.0)
                if "cont" in row:
                    first[g][f"{m}_c"] = float(row["cont"])
        second = {g: {"s3": float((run2_games.get(g) or {}).get("score") or 0.0),
                      **({"s3_c": float(run2_games[g]["cont"])} if "cont" in (run2_games.get(g) or {}) else {})}
                  for g in self.pool}
        self.state["references"] = []
        self.state.pop("reference", None)
        self.add_reference(first)
        self.add_reference(second)
        self.state["incumbent"] = cid
        logs = item.get("pool_logs") or []
        self.event_log({"event": "rsi2_incumbent", "candidate": cid})
        if logs and Path(logs[0]).exists():
            class _Run:  # the first pool run's files
                actions = Path(logs[0])
                snapshots = Path(logs[0]).with_name("snapshots.jsonl")
            await self.frontier_levels(cid, _Run)
        else:
            self.state["frontier"] = {"incumbent": cid, "phase": "failed"}

    async def run_frontier_job(self, job, item):
        from . import frontier as F

        fr = self.state["frontier"]
        cid = job["candidate"]
        keys = self.frontier_instances(job)
        if not keys:
            if job["kind"] == "frontier":  # no frontier set: the full pool decides
                self.state["jobs"].insert(0, {"id": self.new_id("j"), "kind": "eval", "candidate": cid,
                                              "queued": time.time()})
            return
        specs = fr["specs"]
        exposed = [f"k{i:03d}" for i in range(len(keys))]
        wall = len(keys) * self.cfg["frontier_attempt_seconds"]
        prefixes = {e: specs[k] for e, k in zip(exposed, keys)}
        run = Run(self, cid, self.candidate_dir(cid), kind=job["kind"], games=[specs[k]["source"] for k in keys],
                  games_dir=self.root / "games/pool/environment_files", wall=wall, tag=f"{cid}-{job['id']}",
                  prefixes=prefixes)
        result = await run.go()
        first, void = run.first_clears({e: specs[k]["level"] for e, k in zip(exposed, keys)},
                                       result.get("started") or time.time(), wall)
        attempts = {}
        for e, k in zip(exposed, keys):
            if e not in void:
                attempts.setdefault(k, []).append(F.attempt_stats(first[e], wall))
        record = {"job": job["id"], "kind": job["kind"], "wall": wall, "void": sorted(void),
                  "levels": {k: {"attempts": len(v), "clear_rate": round(sum(x["y"] for x in v) / len(v), 3),
                                 "speed": round(sum(x["c"] for x in v) / len(v), 2)} for k, v in attempts.items()},
                  "status": result["status"]}
        if job["kind"] == "fsel":
            rates = {k: v["clear_rate"] for k, v in record["levels"].items()}
            fr.update(selected=F.pick_levels(rates, self.cfg["frontier_k"]), selection=record, phase="need_reference")
            self.state["jobs"].insert(0, {"id": self.new_id("j"), "kind": "fref", "candidate": cid,
                                          "queued": time.time()})
        elif job["kind"] == "fref":
            fr.update(ref=attempts, reference_record=record, phase="ready")
        else:
            z = F.frontier_z(attempts, fr.get("ref") or {})
            ok = bool(z) and z["z"] >= self.cfg["frontier_z_accept"] and z["diff_clear"] > 0 and z["diff_speed"] > 0
            item["frontier"] = {**record, "vs_incumbent": fr["incumbent"], "z": z, "pass": ok}
            item["status"] = "frontier pass, full pool queued" if ok else "done"
            item["verdict"] = None if ok else f"frontier reject (z {z['z'] if z else 'n/a'})"
            if ok:
                self.state["jobs"].insert(0, {"id": self.new_id("j"), "kind": "eval", "candidate": cid,
                                              "queued": time.time()})
            self.write_scores(cid)
            self.notify({"event": "frontier_done", "candidate": cid, "pass": ok, "z": z})
        self.event_log({"event": "rsi2_frontier_job", "job": job["id"], "kind": job["kind"], "candidate": cid,
                        "levels": len(record["levels"]), "void": len(void),
                        **({"z": item.get("frontier", {}).get("z")} if job["kind"] == "frontier" else {})})
        self.state["runs"].append({k: v for k, v in result.items() if k != "marks"})

    def notify(self, row):
        if getattr(self, "proposer", None) is not None:
            self.proposer.notify(row)

    def launch(self):
        if self.job_task is not None and self.job_task.done():
            self.job_task = None
        if not self.state.get("reference") and self.job_task is None and not any(
                j["candidate"] == "c0000" and j["kind"] == "eval" for j in self.state["jobs"]):
            if self.state.get("reference_failures", 0) >= 2:
                if not self.stopping:
                    self.event_log({"event": "rsi2_reference_failed_twice"})
                self.stopping = True
                return
            self.state["jobs"].insert(0, {"id": self.new_id("j"), "kind": "eval", "candidate": "c0000",
                                          "queued": time.time()})
        if (self.dispatcher is not None and self.cfg["split_mode"] == "halves"
                and not self.state.get("split_refs_queued")):
            self.state["split_refs_queued"] = True  # gen 0 measured split, twice (reference + noise check)
            have = sum(1 for j in self.state["jobs"] if j["candidate"] == "c0000" and j["kind"] == "eval")
            for _ in range(max(0, int(self.cfg["split_refs"]) - have)):
                self.state["jobs"].insert(0, {"id": self.new_id("j"), "kind": "eval", "candidate": "c0000",
                                              "queued": time.time()})
            self.state["cont_ref_queued"] = True
        fr = self.state.setdefault("frontier", {"incumbent": "c0000", "phase": None})
        if (self.cfg["frontier"] and self.state.get("reference") and fr["phase"] is None
                and fr["incumbent"] == "c0000"):  # a logged gen-0 pool run for the frontier levels
            fr["phase"] = "need_log"
            if not any(j["candidate"] == "c0000" and j["kind"] == "eval" for j in self.state["jobs"]):
                self.state["cont_ref_queued"] = True  # the same run is the cont reference
                self.state["jobs"].insert(0, {"id": self.new_id("j"), "kind": "eval", "candidate": "c0000",
                                              "queued": time.time()})
        if (self.cfg["continuous"] and self.state.get("reference") and not self.ref_n("s3_c")
                and not self.state.get("cont_ref_queued")):  # gen 0's cont reference (a leg resumed from l2)
            self.state["cont_ref_queued"] = True
            self.state["jobs"].insert(0 if self.job_task is None else 0,
                                      {"id": self.new_id("j"), "kind": "eval", "candidate": "c0000", "queued": time.time()})
        if self.job_task is not None or not self.state["jobs"] or self.stopping:
            return
        i = self.local_index()
        if i is None:
            return
        job = self.state["jobs"][i]
        if job["kind"] == "practice" and int(self.cfg.get("practice_parallel", 1)) > 1:
            picks = self.practice_batch(i)
            batch = [self.state["jobs"][k] for k in picks]
            if time.time() + max(float(j["seconds"]) for j in batch) + 2400 > self.deadline:
                return
            for k in reversed(picks):
                self.state["jobs"].pop(k)
            self.current_job = {"kind": "batch", "id": "+".join(j["id"] for j in batch), "jobs": batch}
            for j in batch:
                self.event_log({"event": "rsi2_job_start", "batch": len(batch),
                                **{k: j.get(k) for k in ("id", "kind", "candidate")}})
            self.job_task = asyncio.create_task(self.run_batch(batch))
            return
        need = (self.pool_wall() if job["kind"] in ("eval", "confirm") else
                self.frontier_seconds(job) if job["kind"] in ("frontier", "fsel", "fref") else job["seconds"])
        if time.time() + need + 2400 > self.deadline:
            return  # it would not finish in this leg; it stays queued for the next
        self.state["jobs"].pop(i)
        self.current_job = job
        self.event_log({"event": "rsi2_job_start", **{k: job.get(k) for k in ("id", "kind", "candidate")}})
        self.job_task = asyncio.create_task(self.run_job(job))

    # ----- setup and main loop ----------------------------------------------------------------------------------
    async def setup(self):
        await asyncio.to_thread(self.runtime.prepare)
        await self.proxy.start()
        for base in (self.ckpt / "mirror", self.source / "configs/rsi2_resume/mirror"):  # a relaunch
            for rel in ("archive", "meta", "prop/ws/work", "prop/home"):
                if (base / rel).is_dir():
                    for item in (base / rel).iterdir():
                        if (rel, item.name) in (("archive", "c0000"), ("meta", "m0000")):
                            continue  # gen 0's tree and setup always come from the bundle (mirrors leave out files
                            # > 32 MB; the operator's setup may be newer than the checkpoint's)
                        if not (self.root / rel / item.name).exists():
                            (self.root / rel).mkdir(parents=True, exist_ok=True)
                            shutil.copytree(item, self.root / rel / item.name, symlinks=True) if item.is_dir() \
                                else shutil.copy2(item, self.root / rel / item.name)
            if (base / "prop-events.jsonl").is_file() and not (self.root / "prop/ws/events.jsonl").exists():
                (self.root / "prop/ws").mkdir(parents=True, exist_ok=True)
                shutil.copy2(base / "prop-events.jsonl", self.root / "prop/ws/events.jsonl")
        gen0 = self.candidate_dir("c0000")
        if not gen0.exists():
            shutil.copytree(self.source / "rsi/gen0/tree", gen0, symlinks=False)
            self.runtime.seal(gen0.parent)
        for cid in self.state["candidates"]:
            if cid != "c0000" and self.candidate_dir(cid).is_dir():
                self.restore_big_files(cid)
            if (self.root / "archive" / cid).is_dir():
                self.runtime.seal(self.root / "archive" / cid)
            if (self.state["candidates"].get("c0000") or {}).get("marks"):
                self.write_scores("c0000")  # a resumed reference: gen 0's scores.json for the proposer
        if "c0000" not in self.state["candidates"]:  # its reference run is the first job (Loop.launch)
            self.state["candidates"]["c0000"] = {
                "parent": None, "note": "gen 0: v8.5 with every public-25 fit refitted on the 10 fixed games",
                "author": "operator", "meta": None, "created": time.time(), "serving_hash": cand.serving_hash(gen0),
                "problems": [], "status": "reference-pending", "verdict": None, "marks": {}}
        self.stacks.trusted_hash = self.state["candidates"]["c0000"]["serving_hash"]
        self.state["stack"].update(live="c0000", live_hash=self.stacks.trusted_hash)  # the bootstrap started gen 0
        self.state["meters"]["legs"] += 1
        self.proposer = ProposerHost(self)
        await self.proposer.setup()
        self.save("setup")

    def restore_big_files(self, cid, limit=32 << 20):
        """The bucket checkpoint leaves out files over 32 MB: a resumed candidate tree gets them back from gen 0, and
        is marked failed if its serving block then differs from what it was submitted with."""
        tree, gen0 = self.candidate_dir(cid), self.source / "rsi/gen0/tree"
        restored = []
        for path in gen0.rglob("*"):
            if path.is_file() and path.stat().st_size > limit:
                rel = path.relative_to(gen0)
                if not (tree / rel).exists():
                    (tree / rel).parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(path, tree / rel)
                    restored.append(rel.as_posix())
        item = self.state["candidates"][cid]
        if restored:
            self.event_log({"event": "rsi2_restored", "candidate": cid, "files": restored})
            if item.get("serving_hash") and cand.serving_hash(tree) != item["serving_hash"] and item.get(
                    "status") in ("queued", "evaluating", "confirming"):
                item.update(status="error", verdict="checkpoint lost a large file of this candidate")
                item["problems"].append("resume: a file over 32 MB was not in the checkpoint and gen 0's differs")
                self.state["jobs"] = [j for j in self.state["jobs"] if j["candidate"] != cid]

    def mirror(self):
        """Checkpoint copies for a relaunch: archive (trees, scores, practice), proposer setups, its work and HOME."""
        for rel in ("archive", "meta", "prop/ws/work", "prop/home"):
            source = self.root / rel
            if source.is_dir():
                target = self.ckpt / "mirror" / rel
                shutil.copytree(source, target, symlinks=True, dirs_exist_ok=True,
                                ignore=shutil.ignore_patterns("node_modules", "__pycache__", ".cache"))
        if (self.root / "prop/ws/events.jsonl").is_file():
            shutil.copy2(self.root / "prop/ws/events.jsonl", self.ckpt / "mirror/prop-events.jsonl")

    async def run(self):
        await self.setup()
        self.stopping = False
        last_save = last_mirror = time.time()
        try:
            while True:
                if not self.stopping and (self.stop_requested() or self.loop_hours() >= self.cfg["budget_loop_gpu_hours"]
                                          or time.time() > self.deadline - self.cfg["end_margin_seconds"]):
                    self.stopping = True
                    self.event_log({"event": "rsi2_stopping", "loop_hours": round(self.loop_hours(), 2)})
                self.meter_worker()
                self.launch()
                self.launch_remote()
                await self.proposer.tick(stop=self.stopping)
                if (self.stopping and self.job_task is None and (self.remote_task is None or self.remote_task.done())
                        and self.proposer.process is None):
                    break
                if time.time() > self.deadline - 300:
                    break
                if self.dispatcher is not None and time.time() - getattr(self, "_last_beat", 0) > 30:
                    self.dispatcher.beat()
                    self._last_beat = time.time()
                if time.time() - getattr(self, "_last_gpu", 0) > 60:
                    self._last_gpu = time.time()
                    await asyncio.to_thread(sample_gpu, self.output / "gpu.jsonl")
                if time.time() - last_save > 60:
                    self.save("tick")
                    last_save = time.time()
                if time.time() - last_mirror > self.cfg["mirror_seconds"]:
                    await asyncio.to_thread(self.mirror)
                    last_mirror = time.time()
                await asyncio.sleep(5)
        finally:
            await self.proposer.stop()
            if self.job_task is not None and not self.job_task.done():
                self.job_task.cancel()
            if self.remote_task is not None and not self.remote_task.done():
                self.remote_task.cancel()
                try:  # its job goes back to the queue (run_job) before the checkpoint below
                    await asyncio.wait_for(asyncio.shield(self.remote_task), 60)
                except BaseException:  # noqa: BLE001
                    pass
            self.state["meters"]["loop_s"] += time.time() - self.started
            self.started = time.time()
            await asyncio.to_thread(self.mirror)
            self.save("end")
            await self.proxy.stop()
        return {"status": "complete", "loop_hours": round(self.loop_hours(), 3),
                "candidates": len(self.state["candidates"]), "accepted": self.state["accepted"]}


async def run_loop(study, trial, **kwargs):
    return await Loop(study, trial, **kwargs).run()
