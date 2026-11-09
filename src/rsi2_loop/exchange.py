"""Split pool evaluations across two VMs: the loop VM plays one half of the pool, a worker VM
(no proposer) the other half, both at the same per-game share (wall = games x slice), and the rule reads the merged
per-game scores at each mark. Fixed side on both machines.

Transport: the Nebius runner copies each run's output folder to the bucket every ~60 s, and the peer's
``rsi-exchange/out`` folder into ``<output>/exchange-in/<peer run>/rsi-exchange/out`` every sync (study key
``nebius_exchange_in``). So a file appears at the peer one to three minutes after it is written, files can arrive
in any order, and a JSON file can be momentarily absent or (in tests) truncated. The protocol therefore:

- writes every file atomically (temporary name, then rename) and every JSON with a ``sha`` of what it refers to;
- never trusts arrival order: a job is taken only when its tree delta has arrived with the stated hash;
- treats every read as optional (``None`` until a valid file is there), never raises on a bad file;
- re-sends: the runner uploads with ``--update`` against the bucket's upload time, so a file rewritten while its
  previous version uploads can be skipped; both sides bump the times of files the peer still needs at every heartbeat
  (the loop: unstarted jobs and their trees, cancels; the worker: its recent results);
- caches what it has read (heartbeat time, marks), so a momentarily unreadable file never undoes progress;
- is idempotent: a job id is run once (a worker that restarts reruns only jobs with no final result);
- has liveness: the worker writes a heartbeat every ``HEARTBEAT`` s; the loop declares it dead after ``DEAD`` s of
  silence, or when a job's progress lags its own marks by ``LATE`` s, and then plays that half itself.

Layout (``out`` = my folder, ``peer`` = my copy of the other side's):
  loop    out/jobs/<jid>.json, out/trees/<jid>.tar.gz, out/cancel/<jid>.json, out/loop.json (heartbeat)
  worker  out/hb.json, out/results/<jid>/progress.json, out/results/<jid>/final.json (last)
"""
from __future__ import annotations

import hashlib
import io
import json
import os
import shutil
import tarfile
import time
from pathlib import Path

HEARTBEAT = 30
DEAD = 900
LATE = 1200
BOOT = 3600  # a worker never heard from is alive this long after the loop's start (VM boot, setup, first sync)
LOOP_WAIT = 1800  # a worker with no live loop VM for this long ends (l8w idled ~12 h alone)
STARTUP = 1800  # a posted job that has not started yet: time allowed for the worker to finish a job and switch stacks
MAX_TREE = 60 << 20  # the runner's interim sync skips files over 64 MB: a bigger delta is played here instead
RESEND = 5  # recent result folders whose files the worker re-sends with every heartbeat


def sha_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha_file(path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


class Mailbox:
    def __init__(self, out, peer):
        self.out, self.peer = Path(out), Path(peer)
        self.out.mkdir(parents=True, exist_ok=True)
        self.lock()

    def lock(self):
        """Root only: the sandboxed player must not read other jobs' trees or scores (the runner creates
        ``exchange-in`` after our start, so this runs at every heartbeat)."""
        for path in (self.out.parent, *[p for p in self.peer.parents if p.name == "exchange-in"]):
            try:
                if path.exists() and path.stat().st_mode & 0o077:
                    os.chmod(path, 0o700)
            except OSError:
                pass

    def write(self, rel, data: bytes):
        path = self.out / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_name(f".{path.name}.tmp{os.getpid()}")
        tmp.write_bytes(data)
        os.replace(tmp, path)

    def write_json(self, rel, obj):
        self.write(rel, json.dumps(obj, sort_keys=True).encode())

    def read_json(self, rel, mine=False):
        try:
            data = json.loads(((self.out if mine else self.peer) / rel).read_text())
            return data if isinstance(data, dict) else None
        except (OSError, ValueError):
            return None

    def peer_path(self, rel):
        return self.peer / rel

    def touch(self, rel):
        try:
            os.utime(self.out / rel)
        except OSError:
            pass

    def list_peer(self, rel, suffix=".json"):
        folder = self.peer / rel
        try:
            return sorted(p.name[:-len(suffix)] for p in folder.iterdir()
                          if p.name.endswith(suffix) and not p.name.startswith("."))
        except OSError:
            return []


# ----- candidate trees as a delta against gen 0 (the worker has gen 0's tree in its own bundle) -----------------------
def tree_files(root):
    root = Path(root)
    out = {}
    for path in sorted(root.rglob("*")):
        if path.is_file() and not path.is_symlink() and "__pycache__" not in path.parts:
            out[path.relative_to(root).as_posix()] = sha_file(path)
    return out


def pack_delta(tree, base):
    """(tar.gz bytes, manifest): the files of ``tree`` that differ from ``base``; the manifest lists every file."""
    files, ref = tree_files(tree), tree_files(base)
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:gz") as tar:
        for rel, digest in files.items():
            if ref.get(rel) != digest:
                tar.add(Path(tree) / rel, arcname=rel)
    return buf.getvalue(), files


def unpack_delta(data, manifest, base, target):
    """``target`` = ``base`` minus files not in the manifest, plus the delta; every file checked against the manifest."""
    target = Path(target)
    if target.exists():
        shutil.rmtree(target)
    shutil.copytree(base, target, symlinks=False, ignore=shutil.ignore_patterns("__pycache__"))
    for rel in list(tree_files(target)):
        if rel not in manifest:
            (target / rel).unlink()
    with tarfile.open(fileobj=io.BytesIO(data), mode="r:gz") as tar:
        for member in tar.getmembers():
            if not member.isfile() or member.name.startswith("/") or ".." in Path(member.name).parts:
                raise ValueError(f"bad member {member.name}")
        tar.extractall(target, filter="data")
    got = tree_files(target)
    if got != manifest:
        bad = sorted(set(got.items()) ^ set(manifest.items()))[:5]
        raise ValueError(f"tree does not match its manifest: {bad}")
    return target


def split_games(pool):
    """Two halves, alternating (pool order), so each half spans the pool."""
    return list(pool[0::2]), list(pool[1::2])


def half_plan(games, slice_seconds, fractions):
    wall = round(len(games) * slice_seconds)
    return wall, {m: round(wall * f) for m, f in fractions.items()}


# ----- the loop side ---------------------------------------------------------------------------------------------------
class Dispatcher:
    """Posts a half to the worker and follows it. ``now`` is injectable for tests."""

    def __init__(self, mailbox, base_tree, now=time.time):
        self.box, self.base, self.now = mailbox, Path(base_tree), now
        self.started = now()
        self.last_hb = None
        self.seen = {}  # jid -> {"started": t, "marks": {...}, "final": doc}
        self.open, self.cancels = set(), set()

    def post(self, jid, cid, tree, games, wall, marks, extra=None):
        """Post a job; False (nothing posted) when its tree delta is too big for the interim sync. ``extra``: more job
        fields (a practice run: kind, batch id, batch size, memory share)."""
        data, manifest = pack_delta(tree, self.base)
        if len(data) > MAX_TREE:
            return False
        self.open.add(jid)
        self.box.write(f"trees/{jid}.tar.gz", data)
        self.box.write(f"trees/{jid}.manifest.json", json.dumps(manifest, sort_keys=True).encode())
        self.box.write_json(f"jobs/{jid}.json", {
            **(extra or {}), "jid": jid, "cid": cid, "games": list(games), "wall": wall, "marks": marks,
            "posted": self.now(),
            "tree_sha": sha_bytes(data),
            "manifest_sha": sha_bytes(json.dumps(manifest, sort_keys=True).encode())})
        return True

    def cancel(self, jid, why):
        self.cancels.add(jid)
        self.box.write_json(f"cancel/{jid}.json", {"jid": jid, "why": why, "t": self.now()})

    def beat(self):
        self.box.lock()
        self.box.write_json("loop.json", {"t": self.now()})
        for jid in list(self.open):  # re-send what the worker may still need
            self.read(jid)
            if self.seen.get(jid, {}).get("final") or jid in self.cancels and self.seen.get(jid, {}).get("started"):
                self.open.discard(jid)
                continue
            if not self.seen.get(jid, {}).get("started"):
                for rel in (f"jobs/{jid}.json", f"trees/{jid}.tar.gz", f"trees/{jid}.manifest.json"):
                    self.box.touch(rel)
        for jid in self.cancels:
            if not self.seen.get(jid, {}).get("final"):
                self.box.touch(f"cancel/{jid}.json")

    def worker_alive(self):
        hb = self.box.read_json("hb.json")
        if hb is not None:
            try:
                self.last_hb = max(self.last_hb or 0.0, float(hb.get("t", 0)))
            except (TypeError, ValueError):
                pass
        if self.last_hb is None:  # never heard from: alive until BOOT after our own start (boot, setup, first sync)
            return self.now() - self.started < BOOT
        return self.now() - self.last_hb < DEAD

    def read(self, jid):
        """Merge what the worker's files say about ``jid`` into the cache; returns the cache entry."""
        entry = self.seen.setdefault(jid, {"started": None, "marks": {}, "final": None})
        for name in ("progress", "final"):
            doc = self.box.read_json(f"results/{jid}/{name}.json")
            if not doc:
                continue
            if doc.get("started") and not entry["started"]:
                entry["started"] = doc["started"]
            for mark, scores in (doc.get("marks") or {}).items():
                if isinstance(scores, dict):
                    entry["marks"].setdefault(mark, scores)
            if name == "final":
                entry["final"] = doc
        return entry

    def final(self, jid):
        return self.read(jid)["final"]

    def mark(self, jid, name):
        """The worker's per-game scores at mark ``name`` of job ``jid``, or None."""
        return self.read(jid)["marks"].get(name)

    def lagging(self, jid, name, seconds, posted):
        """True when mark ``name`` (at ``seconds`` of the job's own clock) is overdue by LATE after the job's start as
        the worker reports it; before it starts, after its posting time plus STARTUP (and the worker's boot)."""
        started = self.read(jid)["started"]
        if started:
            return self.now() > float(started) + seconds + LATE
        origin = max(posted, self.started + BOOT if self.last_hb is None else posted)
        return self.now() > origin + STARTUP + seconds + LATE


def merge_marks(local, remote):
    """Per-game scores of both halves at one mark: {"games": {...}, "t": max}."""
    games = dict((local or {}).get("games") or {})
    games.update((remote or {}).get("games") or {})
    return {"games": games, "t": max(float((local or {}).get("t") or 0), float((remote or {}).get("t") or 0))}


# ----- the worker side -------------------------------------------------------------------------------------------------
class Worker:
    """Takes jobs in posting order and runs each with ``await run_job(job, tree, report, cancelled) -> final dict``;
    ``heartbeat()`` (a task) keeps the heartbeat going while a job runs."""

    def __init__(self, mailbox, base_tree, work, run_job, now=time.time, log=print):
        self.box, self.base, self.work, self.run_job, self.now, self.log = (mailbox, Path(base_tree), Path(work),
                                                                            run_job, now, log)
        self.work.mkdir(parents=True, exist_ok=True)
        self.running = []

    def beat(self):
        self.box.lock()
        finals = sorted((self.box.out / "results").glob("*/final.json"), key=lambda p: p.stat().st_mtime)
        self.box.write_json("hb.json", {"t": self.now(), "running": self.running,
                                        "done": [p.parent.name for p in finals][-50:]})
        recent = [p.parent.name for p in finals][-RESEND:] + list(self.running)
        for jid in recent:  # re-send results the loop may have missed
            for name in ("progress", "final"):
                self.box.touch(f"results/{jid}/{name}.json")

    def loop_alive(self, since, wait=LOOP_WAIT):
        """The loop VM has beaten within ``wait`` (or ``since`` is less than ``wait`` ago when it never has)."""
        doc = self.box.read_json("loop.json")
        try:
            last = float(doc.get("t", 0)) if doc else None
        except (TypeError, ValueError, AttributeError):
            last = None
        return self.now() - (last if last else since) < wait

    def pending(self):
        jobs = []
        for jid in self.box.list_peer("jobs"):
            job = self.box.read_json(f"jobs/{jid}.json")
            if not job or job.get("jid") != jid:
                continue
            if (self.box.out / f"results/{jid}/final.json").exists():
                continue
            jobs.append(job)
        return sorted(jobs, key=lambda j: (j.get("posted", 0), j["jid"]))

    def cancelled(self, jid):
        return self.box.read_json(f"cancel/{jid}.json") is not None

    def tree_for(self, job):
        """The job's tree once its delta and manifest have arrived intact, else None (not yet)."""
        tar = self.box.peer_path(f"trees/{job['jid']}.tar.gz")
        man = self.box.peer_path(f"trees/{job['jid']}.manifest.json")
        try:
            data, mdata = tar.read_bytes(), man.read_bytes()
        except OSError:
            return None
        if sha_bytes(data) != job["tree_sha"] or sha_bytes(mdata) != job["manifest_sha"]:
            return None
        return unpack_delta(data, json.loads(mdata), self.base, self.work / job["jid"] / "tree")

    def report(self, jid, doc):
        self.box.write_json(f"results/{jid}/progress.json", {"jid": jid, **doc})

    async def heartbeat(self, stop):
        import asyncio

        while not stop.is_set():
            self.beat()
            try:
                await asyncio.wait_for(stop.wait(), HEARTBEAT)
            except asyncio.TimeoutError:
                pass

    async def run_one(self, job, tree):
        self.running.append(job["jid"])
        self.beat()
        try:
            final = await self.run_job(job, tree, lambda doc: self.report(job["jid"], doc),
                                       lambda: self.cancelled(job["jid"]))
        except Exception as exc:  # noqa: BLE001 - the loop learns of it from the final file
            final = {"status": f"error: {type(exc).__name__}: {exc}"[:500], "marks": {}}
        self.box.write_json(f"results/{job['jid']}/final.json", {"jid": job["jid"], **final})
        self.running.remove(job["jid"])
        self.beat()

    async def step(self):
        """One poll: run the first pending job (in posting order) once its tree is complete, or the whole batch it
        belongs to (practice runs that play at once) once every member has arrived. True if anything ran."""
        import asyncio

        self.beat()
        pending = self.pending()
        for job in pending:
            if self.cancelled(job["jid"]):
                self.box.write_json(f"results/{job['jid']}/final.json",
                                    {"jid": job["jid"], "status": "cancelled before start", "marks": {}})
                continue
            group = [j for j in pending if j.get("batch") == job["batch"]] if job.get("batch") else [job]
            if len(group) < int(job.get("batch_n") or 1):
                return False  # the rest of the batch has not arrived yet
            trees = []
            for member in group:
                try:
                    tree = self.tree_for(member)
                except (ValueError, OSError, tarfile.TarError) as exc:
                    self.box.write_json(f"results/{member['jid']}/final.json",
                                        {"jid": member["jid"], "status": f"bad tree: {exc}"[:300], "marks": {}})
                    return True
                if tree is None:
                    return False  # not arrived yet: wait for it (posting order is kept)
                trees.append(tree)
            await asyncio.gather(*(self.run_one(m, t) for m, t in zip(group, trees)))
            return True
        return False
