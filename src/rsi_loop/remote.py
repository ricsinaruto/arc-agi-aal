"""Scoring on more than one GPU (two loop GPUs, pool games in parallel across both).

The loop VM stays the only controller. A scoring VM (``role`` "scorer") runs the same player stack with no proposer,
no policy and no held-out games: it plays the pool and practice games the loop hands it, on the loop's live stack.
Neither trial has a network; the runner moves files through the bucket (spec ``exchange_in``, about a minute each way):

    loop    <output>/rsi-exchange/out/control.json          live stack, dispatching, end, cancelled job ids
            <output>/rsi-exchange/out/trees/<cid>.tar.gz     candidate trees (once each)
            <output>/rsi-exchange/out/jobs/<scorer>/<id>.json one game: the job, its wall, token cap, concurrency
    scorer  <output>/rsi-exchange/out/status.json           heartbeat: machine and idle GPU-s, live stack, running
            <output>/rsi-exchange/out/results/<id>.json      the game's row (practice: <id>.trace.tar.gz first)

Each side reads the other's folder under <output>/exchange-in/<run id>/rsi-exchange/out. A scorer's machine time is
loop time (GPU-s summed over the scoring GPUs); its idle slot share joins the loop's idle meter."""
from __future__ import annotations

import asyncio
import json
import os
import shutil
import tarfile
import time
import traceback
from pathlib import Path

from .attempt import SLOTS, Attempt

STALE = 900          # a scorer heartbeat older than this: no new games for it; its running games are given up
CONTROL_STALE = 1800  # a scorer whose loop control is older than this ends


def exchange_root(cfg):
    base = cfg.get("exchange_root") or os.environ.get("ARC_VALIDATION_OUTPUT") or "."
    return Path(base)


def out_dir(cfg):
    folder = exchange_root(cfg) / "rsi-exchange/out"
    folder.mkdir(parents=True, exist_ok=True)
    return folder


def inbox(cfg, peer):
    return exchange_root(cfg) / "exchange-in" / peer / "rsi-exchange/out"


def read_json(path):
    try:
        return json.loads(Path(path).read_text())
    except (OSError, ValueError):
        return None


def write_json(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(data))
    os.replace(tmp, path)


def pack(source, target):
    """``source`` folder as a gzip tar at ``target`` (written beside, then moved into place)."""
    target = Path(target)
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp = target.with_name(target.name + ".part")
    with tarfile.open(tmp, "w:gz", compresslevel=3) as tar:
        tar.add(str(source), arcname=".")
    os.replace(tmp, target)
    return target.stat().st_size


def unpack(archive, target):
    """Extract with the ``data`` filter (no absolute paths, no links out, no devices); files the proposer wrote stay
    untrusted on this side too (sealed by the caller)."""
    target = Path(target)
    tmp = target.with_name(target.name + ".unpack")
    shutil.rmtree(tmp, ignore_errors=True)
    tmp.mkdir(parents=True)
    with tarfile.open(archive, "r:gz") as tar:
        tar.extractall(tmp, filter="data")
    shutil.rmtree(target, ignore_errors=True)
    os.replace(tmp, target)


# ----- loop side ---------------------------------------------------------------------------------------------------
class RemoteAttempt:
    """A game handed to a scorer: ``run`` waits for its row (the same row a local Attempt returns)."""

    def __init__(self, loop, job, scorer, spec):
        self.loop, self.job, self.scorer, self.spec = loop, job, scorer, spec

    async def run(self):
        loop, job, cfg = self.loop, self.job, self.loop.cfg
        try:
            if job["candidate"] != "c0000":
                await loop.send_tree(job["candidate"])
            write_json(out_dir(cfg) / "jobs" / self.scorer / f"{job['id']}.json", self.spec)
            box = inbox(cfg, self.scorer) / "results"
            while True:
                await asyncio.sleep(float(cfg.get("remote_poll_seconds", 15)))
                row = read_json(box / f"{job['id']}.json")
                if row is not None:
                    trace = row.pop("trace", None)
                    if trace:
                        await self.take_trace(box / trace["file"], trace["bytes"], trace["name"])
                    return {**row, "scorer": self.scorer}
                status = loop.remote_status(self.scorer)
                age = time.time() - ((status or {}).get("at") or job["started"])
                if age > STALE or time.time() - job["started"] > self.spec["wall"] + 3600:
                    loop.cancel_remote(job["id"])
                    return {"outcome": "error", "infrastructure": True, "scorer": self.scorer,
                            "error": f"scorer {self.scorer}: no result (heartbeat {age:.0f} s old)"}
        except asyncio.CancelledError:
            loop.cancel_remote(job["id"])
            raise

    async def take_trace(self, path, size, name):
        """The practice trace the proposer reads: into the archive view, as a local practice game's."""
        for _ in range(80):  # the row can arrive one sync before a big trace
            if path.is_file() and path.stat().st_size == size:
                break
            await asyncio.sleep(15)
        else:
            return
        view = self.loop.root / "archive" / self.job["candidate"] / "practice" / name
        await asyncio.to_thread(unpack, path, view)
        self.loop.runtime.seal(view)


# ----- scorer side -------------------------------------------------------------------------------------------------
class Scorer:
    def __init__(self, loop):
        self.loop = loop
        self.cfg = loop.cfg
        self.peer = self.cfg["peer"]
        self.me = self.cfg["scorer_id"]
        self.taken = set()
        self.idle_s = 0.0
        self.last = time.time()
        self.trees = {}
        self.started = time.time()
        self.last_control = None

    def control(self):
        """The loop's control; a failed read (the file mid-sync) keeps the last good one."""
        self.last_control = read_json(inbox(self.cfg, self.peer) / "control.json") or self.last_control
        return self.last_control

    def tree(self, cid):
        """The candidate's tree from the loop (c0000 is this bundle's own)."""
        loop = self.loop
        if cid == "c0000" or cid in self.trees:
            return loop.candidate_dir(cid)
        archive = inbox(self.cfg, self.peer) / "trees" / f"{cid}.tar.gz"
        meta = read_json(inbox(self.cfg, self.peer) / "trees" / f"{cid}.json")
        if not meta or not archive.is_file() or archive.stat().st_size != meta["bytes"]:
            return None
        target = loop.candidate_dir(cid)
        unpack(archive, target)
        loop.runtime.seal(target.parent)
        from . import candidates as cand
        loop.state["candidates"][cid] = {**meta["item"], "serving_hash": cand.serving_hash(target), "pool": {},
                                         "pool_levels": {}, "problems": []}
        self.trees[cid] = True
        return target

    def status(self, note=None):
        loop = self.loop
        m = loop.state["meters"]
        write_json(out_dir(self.cfg) / "status.json", {
            "at": time.time(), "scorer": self.me, "machine_s": m["machine_s"], "idle_s": self.idle_s,
            "switch_s": m.get("switch_s", 0.0), "live": loop.state["stack"]["live"],
            "live_hash": loop.state["stack"]["live_hash"], "switching": bool(loop.state.get("switching")),
            "running": sorted(loop.running), "concurrency": loop.state.get("concurrency"),
            "deadline": loop.deadline - loop.cfg["end_margin_seconds"], "ending": loop.stop_launch, "note": note})

    def tick(self):
        loop = self.loop
        loop.tick_meters()
        now = time.time()
        dt, self.last = now - self.last, now
        conc = loop.state.get("concurrency") or 14
        cap = max(1, min(SLOTS, conc))
        if not loop.state.get("switching"):
            self.idle_s += dt * max(0, cap - len(loop.running)) / max(1, conc)

    async def play(self, job, attempt):
        loop = self.loop
        try:
            row = await attempt.run()
        except asyncio.CancelledError:
            return
        except Exception as exc:  # noqa: BLE001
            row = {"outcome": "error", "infrastructure": True, "error": f"{type(exc).__name__}: {exc}"[:500],
                   "traceback": traceback.format_exc()[-2000:]}
        loop.running.pop(job["id"], None)
        box = out_dir(self.cfg) / "results"
        if job["kind"] == "practice":
            name = f"{attempt.point['game']}-{job['id']}"
            view = loop.root / "archive" / job["candidate"] / "practice" / name
            if view.is_dir():
                size = await asyncio.to_thread(pack, view, box / f"{job['id']}.trace.tar.gz")
                row["trace"] = {"file": f"{job['id']}.trace.tar.gz", "bytes": size, "name": name}
        write_json(box / f"{job['id']}.json", row)
        loop.event_log({"event": "rsi_scorer_game", **{k: row.get(k) for k in (
            "outcome", "score", "levels", "tokens", "seconds")}, "id": job["id"], "kind": job["kind"]})

    def start_jobs(self):
        loop = self.loop
        folder = inbox(self.cfg, self.peer) / "jobs" / self.me
        if not folder.is_dir():
            return
        for path in sorted(folder.glob("j*.json")):
            jid = path.stem
            if jid in self.taken:
                continue
            spec = read_json(path)
            if not spec:
                continue
            job = spec["job"]
            tree = self.tree(job["candidate"])
            if tree is None or loop.state["candidates"][job["candidate"]]["serving_hash"] != \
                    loop.state["stack"]["live_hash"]:
                continue  # its tree is still on the way, or the stack follows the loop's first
            conc = int(spec["concurrency"])
            if loop.running and conc != loop.state.get("concurrency"):
                return  # one concurrency per batch, as on the loop
            if len(loop.running) >= min(SLOTS, conc):
                return
            loop.state["concurrency"] = conc
            self.taken.add(jid)
            used = {j.get("slot") for j, _ in loop.running.values()}
            job = {**job, "slot": next(s for s in range(SLOTS) if s not in used), "started": time.time()}
            set_name = "practice" if job["kind"] == "practice" else "pool"
            point = loop.points[set_name][job["game"]]
            attempt = Attempt(loop.runtime, job, point, tree, wall=spec["wall"], meter="loop", games_set=set_name,
                              concurrency=conc, token_cap=spec.get("token_cap"))
            loop.running[jid] = (job, asyncio.create_task(self.play(job, attempt)))

    async def follow_stack(self, control):
        loop = self.loop
        want, want_hash = control.get("live"), control.get("live_hash")
        if not want or want_hash == loop.state["stack"]["live_hash"] or loop.running:
            return
        if self.tree(want) is None:
            return
        if loop.state["candidates"][want]["serving_hash"] != want_hash:
            return
        loop.tick_meters()
        loop.state["switching"] = True
        try:
            record = await loop.switch_stack(want, loop.state["stack"]["live"])
        finally:
            loop.tick_meters()
            loop.state["switching"] = False
        loop.event_log({"event": "rsi_scorer_switch", **{k: record.get(k) for k in ("from", "to", "ok", "error")}})

    async def run(self):
        loop = self.loop
        from .proposer import ProposerHost
        from .isolation import self_test
        await asyncio.to_thread(loop.runtime.prepare)
        await loop.proxy.start()
        loop.proposer = ProposerHost(loop)  # set up (users, folders) for the isolation self-test; never started
        tree = loop.candidate_dir("c0000")
        await asyncio.to_thread(loop.runtime.install_gen0, tree)
        loop.register_candidate("c0000", tree, None, "gen 0", "build")
        loop.state["stack"].update(live="c0000", live_hash=loop.state["candidates"]["c0000"]["serving_hash"])
        loop.stacks.trusted_hash = loop.state["stack"]["live_hash"]
        await loop.proposer.setup()
        await self_test(loop)
        loop.save("start")
        last_status = 0.0
        try:
            while True:
                self.tick()
                control = self.control()
                now = time.time()
                if control is None:
                    if now - self.started > 3 * 3600:
                        loop.event_log({"event": "rsi_scorer_end", "reason": "no control from the loop"})
                        break
                else:
                    for jid in control.get("cancel") or []:
                        if jid in loop.running:
                            job, task = loop.running.pop(jid)
                            task.cancel()
                            loop.event_log({"event": "rsi_scorer_cancelled", "id": jid})
                    if control.get("end") or now - control["at"] > CONTROL_STALE:
                        loop.event_log({"event": "rsi_scorer_end", "reason": "loop ended" if control.get("end")
                                        else "loop control stale"})
                        break
                    if loop.stop_requested() or now > loop.deadline - loop.cfg["end_margin_seconds"]:
                        loop.stop_launch = True
                    if not loop.stop_launch and not loop.state.get("switching"):
                        await self.follow_stack(control)
                        if control.get("dispatching", True):
                            self.start_jobs()
                    if loop.stop_launch and not loop.running:
                        break
                if now - last_status > 30:
                    self.status()
                    last_status = now
                if now - getattr(self, "last_save", 0) > 60:
                    loop.save("meters")
                    self.last_save = now
                if now > loop.deadline - 120:
                    break
                await asyncio.sleep(5)
        finally:
            for job, task in list(loop.running.values()):
                task.cancel()
            loop.running.clear()
            loop.stop_launch = True
            self.status(note="ended")
            if not loop.trusted_live():
                await asyncio.to_thread(loop.stacks.stop)
            loop.save("end")
            await loop.proxy.stop()
        return {"status": "complete", "role": "scorer", "games": len(self.taken)}
