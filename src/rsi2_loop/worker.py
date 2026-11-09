"""The evaluation worker VM of a split loop: no proposer, no state of its own; it plays the
halves the loop posts (rsi2_loop.exchange) on its own GPU, each on that candidate's serving stack, with the same fixed
gateway, sandbox and scorer as the loop VM. Root process inside the bundle's level-bench trial (study ``rsi2_worker``).
"""
from __future__ import annotations

import asyncio
import io
import json
import os
import shutil
import tarfile
import time
from pathlib import Path

from . import candidates as cand
from .serving import sample_gpu
from . import exchange as X
from .evaluate import Run
from .runtime import Runtime
from .serving import Stacks


class _Hold:  # Run holds the proposer's proxy while arena.run starts; a worker has no proposer
    def __init__(self):
        self.open = asyncio.Event()
        self.open.set()
        self.inflight = {}


class WorkerHost:
    def __init__(self, study, trial, *, source, games_dir, output, server_dir, base_url, deadline_epoch, log=print,
                 **_):
        self.cfg = {"root": "/rsi", **(study.get("rsi2_worker") or {})}
        self.source, self.output, self.log = Path(source), Path(output), log
        self.deadline = float(deadline_epoch)
        self.root = Path(self.cfg["root"])
        self.ckpt = self.output / "rsi2w"
        self.ckpt.mkdir(parents=True, exist_ok=True)
        self.events = (self.ckpt / "events.jsonl").open("a")
        self.upstream = base_url.rstrip("/")
        from .loop import Loop

        self.served = study.get("served_model") or Loop.served_from(self.source)
        self.proxy = _Hold()
        self.runtime = Runtime(self.root, self.source, games_dir, self.cfg, self.proxy, self.ckpt)
        self.stacks = Stacks(self.source, server_dir, self.upstream, self.served, self.root / "root/stacks",
                             log=self.event_log, runtime=self.runtime)
        self.gen0 = self.source / "rsi/gen0/tree"
        self.live = ("c0000", cand.serving_hash(self.gen0), self.gen0 / "serving")
        self.dead = None
        self.stacks.trusted_hash = self.live[1]
        self.seq = 0
        self.stack_lock = asyncio.Lock()
        self.box = None

    def event_log(self, row):
        row = {"t": time.time(), **row}
        self.events.write(json.dumps(row, default=str) + "\n")
        self.events.flush()
        self.log(json.dumps(row, default=str), flush=True)

    def new_id(self, prefix):
        self.seq += 1
        return f"{prefix}w{self.seq:04d}"

    def save(self, reason=""):
        pass

    def stop_requested(self):
        local = os.environ.get("ARENA_STOP_REQUEST")
        return any(p.exists() for p in (self.output / "stop-request", self.output.parent.parent / "live/stop-request",
                                        *([Path(local)] if local else [])))

    async def ensure_stack(self, cid, tree):
        async with self.stack_lock:  # the runs of a practice batch share one stack
            return await self._ensure_stack(cid, tree)

    async def _ensure_stack(self, cid, tree):
        h = cand.serving_hash(tree)
        if h == self.live[1]:
            return None
        try:
            record = await self.stacks.switch(tree / "serving", cid, self.live[2], self.live[0],
                                              fallback=(self.gen0 / "serving", "c0000"))
        except Exception as exc:  # nothing came back: no server here any more, this VM ends
            self.dead = f"serving stack down after {cid}: {type(exc).__name__}: {exc}"[:600]
            self.event_log({"event": "rsi2w_stack_down", "candidate": cid, "error": self.dead})
            raise
        if record["live"] == "new":
            # the stack runs from a copy that outlives the job's work folder
            keep = self.root / "root/live-serving" / cid
            if not keep.exists():
                shutil.copytree(tree / "serving", keep, symlinks=False)
            self.live = (cid, h, keep)
        elif record["live"] == "fallback":
            self.live = ("c0000", cand.serving_hash(self.gen0), self.gen0 / "serving")
        self.event_log({"event": "rsi2w_stack", **{k: record.get(k) for k in ("from", "to", "ok", "live", "seconds",
                                                                             "error")}})
        return None if record.get("ok") else f"serving: {record.get('error')}"

    async def run_job(self, job, tree, report, cancelled):
        problem = await self.ensure_stack(job["cid"], tree)
        if problem:
            return {"status": problem[:500], "marks": {}}
        if job.get("kind") == "practice":
            return await self.run_practice(job, tree, report, cancelled)
        run = Run(self, job["cid"], tree, kind="eval", games=job["games"],
                  games_dir=self.root / "games/pool/environment_files", wall=job["wall"], marks=job["marks"],
                  tag=job["jid"], log_actions=True)
        marks = {}

        def decide(mark, scores):
            marks[mark] = scores
            report({"started": run.result.get("started"), "marks": marks})
            return "continue"

        async def watch():
            told = False
            while True:
                await asyncio.sleep(15)
                if not told and run.result.get("started"):  # the loop times its marks from the real start
                    told = True
                    report({"started": run.result["started"], "marks": marks})
                if cancelled():
                    run.stop_request("cancelled by the loop")
                    return

        watcher = asyncio.create_task(watch())
        try:
            report({"started": None, "marks": {}})
            result = await run.go(decide)
        finally:
            watcher.cancel()
        self.event_log({"event": "rsi2w_job", "jid": job["jid"], "status": result["status"], "marks": sorted(marks)})
        return {"status": result["status"], "started": result.get("started"), "ended": result.get("ended"),
                "marks": marks, "cancelled": cancelled()}

    async def run_practice(self, job, tree, report, cancelled):
        """A practice run for the loop: its traces go back as results/<jid>/practice.tar.gz (sha in the final)."""
        run = Run(self, job["cid"], tree, kind="practice", games=job["games"],
                  games_dir=self.root / "games/practice/environment_files", wall=job["wall"], tag=job["jid"])
        run.mem_share = float(job.get("share") or 1.0)

        async def watch():
            told = False
            while True:
                await asyncio.sleep(15)
                if not told and run.result.get("started"):
                    told = True
                    report({"started": run.result["started"], "marks": {}})
                if cancelled():
                    run.stop_request("cancelled by the loop")
                    return

        watcher = asyncio.create_task(watch())
        try:
            result = await run.go()
        finally:
            watcher.cancel()
        final = run.scores_at(job["wall"] + 3600, result.get("started") or time.time())
        data = await asyncio.to_thread(pack_traces, run.out / "duck-transfer/games")
        self.box.write(f"results/{job['jid']}/practice.tar.gz", data)
        self.event_log({"event": "rsi2w_job", "jid": job["jid"], "status": result["status"], "kind": "practice",
                        "trace_bytes": len(data)})
        return {"status": result["status"], "started": result.get("started"), "ended": result.get("ended"),
                "marks": {}, "cancelled": cancelled(),
                "practice": {"scores": (final or {}).get("games"), "sha": X.sha_bytes(data), "bytes": len(data)}}

    async def selftest(self):
        """Before any job: gen 0's trusted stack is stopped and started again (the way back after a failed candidate
        stack must work). With ``selftest_untrusted`` also gen 0's serving as an untrusted candidate first (off while
        that start fails its preflight, 18 Oct). A failure ends this VM before it takes work."""
        started, problems = time.time(), []
        if self.cfg.get("selftest_untrusted"):
            tree = self.root / "root/selftest"
            shutil.rmtree(tree, ignore_errors=True)
            shutil.copytree(self.gen0 / "serving", tree / "serving", symlinks=False)
            (tree / "serving/selftest.txt").write_text("switch-path self-test\n")
            problems += [p for p in [await self._ensure_stack("selftest", tree)] if p]
        if not self.dead:
            try:
                record = await self.stacks.switch(self.gen0 / "serving", "c0000-selftest", self.gen0 / "serving",
                                                  "c0000")
                if not (record.get("ok") and record.get("live") == "new"):
                    problems.append(f"gen-0 restart: {record.get('error')}")
            except Exception as exc:  # noqa: BLE001 - nothing came back
                self.dead = f"gen-0 restart: {type(exc).__name__}: {exc}"[:600]
            self.live = ("c0000", cand.serving_hash(self.gen0), self.gen0 / "serving")
        ok = not problems and not self.dead
        self.event_log({"event": "rsi2w_selftest", "ok": ok, "seconds": round(time.time() - started, 1),
                        "problems": problems[:2], "untrusted": bool(self.cfg.get("selftest_untrusted")),
                        **({"dead": self.dead} if self.dead else {})})
        if not ok:
            self.dead = self.dead or f"switch self-test failed: {problems[:2]}"

    async def run(self):
        await asyncio.to_thread(self.runtime.prepare, True)
        self.dead = None
        if self.cfg.get("stack_selftest", True):
            try:
                await self.selftest()
            except Exception as exc:  # noqa: BLE001
                self.dead = self.dead or f"switch self-test: {type(exc).__name__}: {exc}"[:600]
            if self.dead:
                return {"status": f"error: {self.dead}", "jobs": 0}
        peer = self.cfg["peer"]
        base = Path(os.environ.get("ARC_VALIDATION_OUTPUT") or self.cfg.get("exchange_root") or self.output)
        box = self.box = X.Mailbox(base / "rsi-exchange/out", base / "exchange-in" / peer / "rsi-exchange/out")
        worker = X.Worker(box, self.gen0, self.root / "wjobs", self.run_job, log=self.event_log)
        stop = asyncio.Event()
        beats = asyncio.create_task(worker.heartbeat(stop))

        async def gpu_samples():  # nvidia-smi every minute, jobs running or not
            while not stop.is_set():
                await asyncio.to_thread(sample_gpu, self.output / "gpu.jsonl")
                try:
                    await asyncio.wait_for(stop.wait(), 60)
                except asyncio.TimeoutError:
                    pass
        gpus = asyncio.create_task(gpu_samples())
        jobs = 0
        since = time.time()
        try:
            while time.time() < self.deadline - 600 and not self.stop_requested() and not self.dead:
                if not worker.running and not worker.loop_alive(since):
                    self.dead = f"no loop VM heartbeat for {X.LOOP_WAIT // 60} min ({peer})"
                    self.event_log({"event": "rsi2w_no_loop", "why": self.dead})
                    break
                if await worker.step():
                    jobs += 1
                else:
                    await asyncio.sleep(15)
        finally:
            stop.set()
            await beats
            await gpus
        return {"status": f"error: {self.dead}" if self.dead else "complete", "jobs": jobs}


def pack_traces(folder, limit=None):
    """tar.gz of a practice run's game traces (no databases); the largest files are left out above ``limit``."""
    limit = limit or X.MAX_TREE
    folder = Path(folder)
    files = sorted((p for p in folder.rglob("*") if p.is_file() and not p.is_symlink()
                    and not p.name.endswith((".db", ".sqlite", ".sqlite-wal", ".sqlite-shm"))),
                   key=lambda p: p.stat().st_size) if folder.is_dir() else []
    while True:
        buf = io.BytesIO()
        with tarfile.open(fileobj=buf, mode="w:gz") as tar:
            for path in files:
                tar.add(path, arcname=path.relative_to(folder).as_posix())
        if len(buf.getvalue()) <= limit or not files:
            return buf.getvalue()
        files.pop()  # drop the largest and try again


async def run_worker(study, trial, **kwargs):
    return await WorkerHost(study, trial, **kwargs).run()
