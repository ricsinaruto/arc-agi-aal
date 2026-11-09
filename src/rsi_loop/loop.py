"""The RSI loop controller (docs/rsi.md; record docs/rsi-arm.md). Root process on the loop VM, outside every sandbox:
it owns the game engine (brokers), the scorer, the timer and meters, the held-out games and their results, the
archive, the checkpoint and the serving stack's switches.

Entered through the level-bench trial when the study has an ``rsi_loop`` block (``arena.level_bench``); the
bootstrap has started gen 0's stack before. Work:

- play slots (``concurrency`` of the live candidate, minus one for the proposer): queued games run as Prime Agent
  sessions (``rsi_loop.attempt``), held-out first, then smoke games, pool evaluations, practice;
- one proposer session at a time (``rsi_loop.proposer``), a full Prime Agent session with code execution on its
  workspace; its requests arrive on a socket; its archive view gets pool scores and practice traces only;
- the selection policy (``meta/policy.py`` of the live proposer setup, run in the proposer's sandbox; gen 0's
  ``rsi_loop.policy`` if it fails) queues evaluations after every submission and finished evaluation;
- held-out (docs/rsi.md 4): A with one seed every ``heldout_a_every`` loop GPU-h if the best changed (gen 0 too the
  first time), A and B with three seeds every ``heldout_ab_every`` loop GPU-h; separate meter, never shown.

Meters: machine seconds since start, split each tick by the running sessions: held-out games to the held-out meter,
everything else (pool, practice, smoke, the proposer, idle slots, stack switches) to the loop. Checkpoint:
``<output>/rsi/`` (state.json atomic after every event; candidate trees and practice traces mirrored), copied to the
bucket by the runner every 60 s; ``configs/rsi_resume/`` in a relaunch bundle restores it (running games restart).
"""
from __future__ import annotations

import asyncio
import json
import math
import os
import random
import secrets
import shutil
import stat
import subprocess
import time
import traceback
from pathlib import Path

from . import candidates as cand
from . import policy as gen0_policy
from . import registry
from . import sandbox
from .attempt import SLOTS, Attempt, Runtime
from .proposer import ProposerHost
from .proxy import Proxy
from .serving import Stacks, clip

DEFAULTS = {
    "slice_seconds": 32400 / 110,     # Kaggle runtime / 110 GPU-seconds per game (docs/rsi.md 3)
    "heldout_a_every": 10.0,          # loop GPU-h between one-seed held-out A runs (only if the best changed)
    "heldout_ab_every": None,         # loop GPU-h between three-seed A + B runs (postponed; None = off)
    "heldout_seeds": 3,
    # a held-out game plays at its candidate's own concurrency and ends at 100K
    # generated tokens (no time slice); the wall is only a safety stop. Runs of the old format are not comparable.
    "heldout_format": "tokens100k",
    "heldout_token_cap": 100000,
    "heldout_wall_seconds": 3 * 3600,
    # held-out runs play on their own GPU. "remote": the loop machine only records a request
    # (<checkpoint>/heldout/requests/<run>.json); a held-out machine (``role`` "heldout", built from the loop's
    # checkpoint with ``heldout_requests``) plays it. "local": the loop machine plays it alone (proposer paused).
    "heldout_mode": "remote",
    "role": "loop",                   # "heldout": play ``heldout_requests`` and end (no proposer session, no policy)
    # pool and practice games also play on scoring VMs (``rsi_loop.remote``; their run ids)
    "scorers": [],                    # on a scorer: role "scorer", "peer" (the loop's run id), "scorer_id" (its own)
    "heldout_requests": [],
    # 14 Oct (coordinator): a drain or a held-out pause lets the proposer's running session finish, up to this long
    "proposer_grace_seconds": 2700,
    "end_margin_seconds": 900,
    "drain_seconds": 1200,            # a stack switch waits this long for running games, then restarts the rest
    "smoke_seconds": 240,
    "seed": 20261012,
    "proxy_port": 18080,
    "points": {"pool": "configs/rsi/pool_points.json", "practice": "configs/rsi/practice_points.json",
               "heldout_a": "configs/rsi/heldout_a_points.json", "heldout_b": "configs/rsi/heldout_b_points.json"},
    "root": "/rsi",
    "resume": "configs/rsi_resume",
    "gen0_once": True,                # gen 0 is frozen: its three-seed A + B run is made once and reused
    "best_min_games": 15,             # a held-out candidate has scores on at least one full minibatch (gen 0's 15)
    "budget_loop_gpu_hours": 100,     # this run's approved loop GPU-h (docs/rsi.md 3 caps round 1 at 1,000)
    "round_cap_gpu_hours": 1000,
    "operator_log": "configs/rsi_operator.jsonl",
    "big_inputs": [],                 # relaunch: folders holding an earlier leg's big files (archive/<cid>/..., work/..)
    "paired_minibatch": False,        # owner switch (third audit): paired minibatches as gen 0's selection default
    "pool_rule": "level2",            # random play (3 seeds x 300) may not clear level 2 ("level1" stricter)
    "witness": "practice",            # build record only: every Witness game is practice
}
PRIORITY = {"heldout_a": 0, "heldout_b": 0, "smoke": 1, "pool": 2, "practice": 3}


def mirror(source, target, limit=64 * 1024 ** 2, skip=("node_modules", "__pycache__", ".cache", "kernel-venv"),
           left_out=None):
    """Copy new or changed files of ``source`` under ``target`` (by size and mtime); big files and caches stay out
    (``left_out`` collects the files over ``limit``: they are not restored after a relaunch)."""
    source, target = Path(source), Path(target)
    if not source.exists():
        return 0
    copied = 0
    for root, dirs, files in os.walk(source):
        dirs[:] = [d for d in dirs if d not in skip]
        rel = Path(root).relative_to(source)
        for name in files:
            path = Path(root) / name
            try:
                stat = path.lstat()
            except OSError:
                continue
            if path.is_symlink() or not path.is_file():
                continue
            if stat.st_size > limit:
                if left_out is not None:
                    left_out.append({"path": str(path.relative_to(source)), "bytes": stat.st_size})
                continue
            dest = target / rel / name
            try:
                old = dest.stat()
                if old.st_size == stat.st_size and int(old.st_mtime) == int(stat.st_mtime):
                    continue
            except OSError:
                pass
            dest.parent.mkdir(parents=True, exist_ok=True)
            try:
                shutil.copy2(path, dest)
                copied += 1
            except OSError:
                pass
    return copied


def process_start(pid):
    """Epoch at which process ``pid`` started (Linux /proc); now when unknown."""
    try:
        ticks = int(Path(f"/proc/{pid}/stat").read_text().rsplit(")", 1)[1].split()[19])
        btime = next(int(line.split()[1]) for line in Path("/proc/stat").read_text().splitlines()
                     if line.startswith("btime"))
        return btime + ticks / os.sysconf("SC_CLK_TCK")
    except (OSError, ValueError, IndexError, StopIteration):
        return time.time()


def atomic_write(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(data, indent=1))
    os.replace(tmp, path)


class Loop:
    def __init__(self, study, trial, *, source, games_dir, output, server_dir, base_url, deadline_epoch, log=print,
                 **_):
        self.cfg = {**DEFAULTS, **(study.get("rsi_loop") or {})}
        self.study, self.source, self.output, self.log = study, Path(source), Path(output), log
        self.deadline = float(deadline_epoch)
        self.root = Path(self.cfg["root"])
        self.ckpt = self.output / "rsi"
        self.ckpt.mkdir(parents=True, exist_ok=True)
        self.events = (self.ckpt / "events.jsonl").open("a")
        self.points = {kind: {p["game"]: p for p in json.loads((self.source / rel).read_text())["points"]}
                       for kind, rel in self.cfg["points"].items() if (self.source / rel).is_file()}
        self.served = study["served_model"]
        self.upstream = base_url.rstrip("/")
        self.proxy = Proxy(self.upstream.removesuffix("/v1"), self.served, self.root / "root/proxy",
                           self.cfg["proxy_port"], unix_path=self.root / "psock/proxy.sock")
        self.runtime = Runtime(self.root, self.source, games_dir, self.cfg, self.proxy, self.event_log, self.ckpt)
        self.stacks = Stacks(self.source, server_dir, self.upstream, self.served, self.root / "root/stacks",
                             log=self.event_log, runtime=self.runtime)
        self.state = self.load_state()
        self.bundle_games = {kind: set(points) for kind, points in self.points.items()}  # what a scorer VM has too
        for game in self.state.get("added_games", []):  # games the proposer wrote, restored with the state
            self.points.setdefault(game["role"], {})[game["game"]] = game
        pool_rows = json.loads((self.source / self.cfg["points"]["pool"]).read_text())["points"] \
            if (self.source / self.cfg["points"]["pool"]).is_file() else []
        self.base_pool = sorted(p["game"] for p in pool_rows)
        # public games every evaluated candidate plays (queued by the loop, never cancelled)
        self.fixed_pool = sorted(p["game"] for p in pool_rows if p.get("fixed"))
        self.frontier_base = dict(self.state.get("frontier") or {})  # proxy counters restart with the process
        self.games_dir = games_dir
        try:
            from arc_infra.serving_setup import PORT as server_port
        except Exception:  # noqa: BLE001 - tests without the serving stack
            server_port = int(os.environ.get("RSI_SERVER_PORT", "1234"))
        self.server_port = int(server_port)
        self.unhealthy = 0
        self.running = {}  # job id -> (job, Attempt task)
        self.stop_launch = False
        self.last_tick = time.time()
        # this leg's machine time before the loop: the bootstrap (its process started the server and then this
        # trial) counts as loop time, like any other stack start
        boot = max(0.0, self.last_tick - process_start(os.getppid()))
        meters = self.state["meters"]
        meters["machine_s"] += boot
        meters["loop_s"] += boot
        meters["boot_s"] = meters.get("boot_s", 0.0) + boot

    # ----- state ---------------------------------------------------------------------------------------------
    def load_state(self):
        for path in (self.ckpt / "state.json", self.source / self.cfg["resume"] / "state.json"):
            if path.is_file():
                state = json.loads(path.read_text())
                for job in state.pop("running", []):
                    state["jobs"].insert(0, {k: v for k, v in job.items() if k not in ("slot", "started", "remote")})
                # a new machine: no switch is under way, the bootstrap started gen 0's stack, no batch is running
                state["switching"] = False
                state["heldout_hold"] = None
                state["concurrency"] = None
                state.pop("stop_at", None)
                if (state.get("pending_switch") or {}).get("purpose") in ("scrub", "return"):
                    state["pending_switch"] = None
                state["resumed"] = state.get("resumed", []) + [{"at": time.time(), "from": str(path)}]
                return state
        return {"version": 2, "started": time.time(), "seq": 0,
                "meters": {"loop_s": 0.0, "heldout_s": 0.0, "machine_s": 0.0, "switch_s": 0.0, "idle_slot_s": 0.0,
                           "loop_tokens": 0, "heldout_tokens": 0, "loop_actions": 0, "heldout_actions": 0,
                           "human_s": 0.0},
                "candidates": {}, "meta": {"live": None, "items": {}}, "stack": {"live": None, "history": []},
                "jobs": [], "rows": [], "practice": {}, "pending_switch": None, "proposer": {"sessions": []},
                "heldout": {"runs": [], "accesses": [], "last_a_best": None, "next_a": None, "next_ab": None},
                "frontier": {"non_flash": 0, "egress": {}, "unknown_key": 0}, "operator_log": [],
                "pool_games": sorted(self.points.get("pool", {})),
                "practice_games": sorted(self.points.get("practice", {}))}

    def big_dir(self):
        """Where files too big for the checkpoint go: the run's output folder (the runner copies it to the bucket
        in full every ten minutes; a relaunch gets them back as an input dataset, ``big_inputs``)."""
        base = os.environ.get("ARC_VALIDATION_OUTPUT")
        folder = Path(base) / "rsi-big" if base else self.ckpt / "big"
        folder.mkdir(parents=True, exist_ok=True)
        os.chmod(folder, 0o700)
        return folder

    def keep_big(self, files, source, area):
        """Copies of ``files`` (paths relative to ``source``) under big_dir/area, read without following a link
        (``source`` may be the proposer's folder); unchanged files (size, mtime) are not copied again."""
        target_root = self.big_dir() / area
        for item in files:
            rel = Path(item["path"])
            try:
                cand._parts(rel)
                parent = cand.open_dir(source, rel.parent)
            except (OSError, ValueError):
                continue
            try:
                fd = os.open(rel.name, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK, dir_fd=parent)
            except OSError:
                os.close(parent)
                continue
            os.close(parent)
            try:
                info = os.fstat(fd)
                target = target_root / rel
                if not stat.S_ISREG(info.st_mode):
                    continue
                if target.is_file() and (target.stat().st_ino == info.st_ino or (
                        target.stat().st_size == info.st_size and int(target.stat().st_mtime) == int(info.st_mtime))):
                    continue
                target.parent.mkdir(parents=True, exist_ok=True)
                target.unlink(missing_ok=True)
                try:  # same file system: a hard link of that very inode (linkat on the descriptor, no path)
                    os.link(f"/proc/self/fd/{fd}", target, follow_symlinks=True)
                    continue
                except OSError:
                    pass
                os.set_blocking(fd, True)
                with open(target, "wb") as writer:
                    while chunk := os.read(fd, 1 << 24):
                        writer.write(chunk)
                os.utime(target, (info.st_atime, info.st_mtime))
            except OSError:
                continue
            finally:
                os.close(fd)

    def mirror_all(self, cids=None):
        """Checkpoint copies of what lives outside the state: proposer setups, the proposer's workspace and HOME,
        practice traces of every candidate (candidate trees are mirrored when archived). Runs in a worker thread:
        ``cids`` is the candidate list taken in the loop's thread; the state is not touched here."""
        left_out = []
        copied = mirror(self.root / "meta", self.ckpt / "meta", left_out=left_out)
        work = []
        skip = ("node_modules", "__pycache__", ".cache", "kernel-venv")
        for rel, target, out in (("ws/work", "prop/work", work), ("home", "prop/home", left_out)):
            try:  # the proposer's own folders: read through descriptors, no link followed
                copied += cand.copy_untrusted(self.root / "prop", rel, self.ckpt / target, skip=skip,
                                              limit=64 * 1024 ** 2, left_out=out, changed_only=True)
            except (OSError, ValueError):
                pass
        self.keep_big(work, self.root / "prop/ws/work", "work")
        try:
            (self.ckpt / "prop").mkdir(parents=True, exist_ok=True)
            (self.ckpt / "prop/events.jsonl").write_bytes(cand.read_untrusted(self.root / "prop/ws", "events.jsonl"))
        except (OSError, ValueError):
            pass
        for cid in (cids if cids is not None else list(self.state["candidates"])):
            copied += mirror(self.root / "archive" / cid / "practice", self.ckpt / "archive" / cid / "practice")
        return copied, work, left_out

    async def mirror_now(self):
        copied, work, other = await asyncio.to_thread(self.mirror_all, list(self.state["candidates"]))
        self.state["big_files"] = {"work": work[:500], "not_checkpointed": other[:500]}

    def save(self, reason):
        self.state["saved"] = {"at": time.time(), "reason": reason}
        self.state["running"] = [{**job, "slot": job.get("slot")} for job, _ in self.running.values()]
        atomic_write(self.ckpt / "state.json", self.state)
        self.state.pop("running")

    def event_log(self, row):
        row = {"time": time.time(), **row}
        self.events.write(json.dumps(row) + "\n")
        self.events.flush()
        self.log(json.dumps(row))

    def new_id(self, prefix):
        self.state["seq"] += 1
        return f"{prefix}{self.state['seq']:05d}"

    # ----- meters ------------------------------------------------------------------------------------------------
    def tick_meters(self):
        now = time.time()
        dt, self.last_tick = now - self.last_tick, now
        m = self.state["meters"]
        m["machine_s"] += dt
        if self.state.get("switching"):
            purpose = (self.state.get("pending_switch") or {}).get("purpose", "proposer")
            m["heldout_s" if purpose in ("heldout", "return", "scrub") else "loop_s"] += dt
            m["switch_s"] += dt
            return
        loop_running = any(not job["kind"].startswith("heldout") for job, _ in self.running.values())
        if self.state.get("heldout_hold") == "held" and not loop_running:
            m["heldout_s"] += dt  # held-out exclusive mode: the whole machine is the held-out run's
            return  # (while loop games still drain, the ordinary split below applies)
        slots = self.slots()
        held = sum(1 for job, _ in self.running.values() if job["kind"].startswith("heldout"))
        share = held / max(1, slots + 1)  # the proposer is one more session on the GPU
        m["heldout_s"] += dt * share
        m["loop_s"] += dt * (1 - share)
        m["idle_slot_s"] += dt * max(0, slots - len(self.local_running())) / max(1, slots + 1)
        self.tick_remote()

    def local_running(self):
        return [(j, t) for j, t in self.running.values() if not j.get("remote")]

    def tick_remote(self):
        """Scoring VMs: their machine time is loop time (GPU-s summed over the scoring GPUs), their idle share joins
        the idle meter; read from their heartbeats (a gap is caught up at the next one)."""
        if not self.cfg["scorers"]:
            return
        m = self.state["meters"]
        seen = self.state.setdefault("remote", {}).setdefault("seen", {})
        for scorer in self.cfg["scorers"]:
            status = self.remote_status(scorer)
            if not status:
                continue
            last = seen.setdefault(scorer, {"machine_s": 0.0, "idle_s": 0.0})
            d = float(status["machine_s"]) - last["machine_s"]
            if d > 0:
                m["loop_s"] += d
                m["remote_s"] = m.get("remote_s", 0.0) + d
                m["idle_slot_s"] += max(0.0, float(status.get("idle_s", 0.0)) - last["idle_s"])
                last.update(machine_s=float(status["machine_s"]), idle_s=float(status.get("idle_s", 0.0)))

    def remote_status(self, scorer):
        from .remote import inbox, read_json
        cache = getattr(self, "_remote_cache", None)
        if cache is None:
            cache = self._remote_cache = {}
        hit = cache.get(scorer)
        if hit and time.time() - hit[0] < 5:
            return hit[1]
        status = read_json(inbox(self.cfg, scorer) / "status.json")
        cache[scorer] = (time.time(), status)
        return status

    def fresh_scorers(self):
        from .remote import STALE
        out = []
        for scorer in self.cfg["scorers"]:
            status = self.remote_status(scorer)
            if status and time.time() - status["at"] < STALE and not status.get("ending"):
                out.append((scorer, status))
        return out

    def scoring_slots(self):
        """Play slots across the scoring GPUs: this machine's plus each live scorer's (it has no proposer: C games)."""
        return self.slots() + sum(min(SLOTS, self.concurrency()) for _ in self.fresh_scorers())

    def cancel_remote(self, jid):
        cancel = self.state.setdefault("remote", {}).setdefault("cancel", [])
        if jid not in cancel:
            cancel.append(jid)
        self.write_control()

    def write_control(self, end=False):
        if not self.cfg["scorers"]:
            return
        from .remote import out_dir, write_json
        write_json(out_dir(self.cfg) / "control.json", {
            "at": time.time(), "live": self.state["stack"]["live"], "live_hash": self.state["stack"]["live_hash"],
            "dispatching": not self.stop_launch and not self.state.get("switching"), "end": end,
            "cancel": (self.state.get("remote") or {}).get("cancel", [])[-500:]})

    async def send_tree(self, cid):
        """A candidate's tree for the scorers (once; trees do not change after archiving)."""
        from .remote import out_dir, pack, write_json
        folder = out_dir(self.cfg) / "trees"
        if (folder / f"{cid}.json").is_file():
            return
        locks = self.__dict__.setdefault("_tree_locks", {})
        lock = locks.setdefault(cid, asyncio.Lock())
        async with lock:
            if (folder / f"{cid}.json").is_file():
                return
            size = await asyncio.to_thread(pack, self.candidate_dir(cid), folder / f"{cid}.tar.gz")
            item = self.state["candidates"][cid]
            write_json(folder / f"{cid}.json", {"bytes": size, "item": {k: item.get(k) for k in (
                "parent", "status", "concurrency", "game_seconds", "game_token_cap", "serving_hash")}})

    def dispatch_remote(self):
        """Pool and practice games for the scoring VMs on the live stack, one concurrency per batch, as here."""
        if not self.cfg["scorers"] or self.heldout_exclusive() or self.state.get("budget_reached"):
            return
        from .remote import RemoteAttempt
        live = self.state["stack"]["live_hash"]
        for scorer, status in self.fresh_scorers():
            if status.get("live_hash") != live:
                if self.state["stack"]["live"] != "c0000":
                    asyncio.create_task(self.send_tree(self.state["stack"]["live"]))  # for its switch
                continue
            mine = [j for j, _ in self.running.values() if j.get("remote") == scorer]
            limit = min(self.deadline, float(status.get("deadline") or self.deadline)) - time.time()
            while True:
                conc = self.job_concurrency(mine[0]) if mine else None
                ready = [j for j in self.state["jobs"] if j["kind"] in ("pool", "practice")
                         and self.state["candidates"].get(j["candidate"], {}).get("serving_hash") == live
                         and j["game"] in self.bundle_games.get("practice" if j["kind"] == "practice" else "pool", ())
                         and self.job_wall(j) <= limit - self.cfg["end_margin_seconds"]
                         and (conc is None or self.job_concurrency(j) == conc)]
                if not ready:
                    break
                job = min(ready, key=lambda j: (PRIORITY[j["kind"]], j["queued"]))
                c = self.job_concurrency(job)
                if len(mine) >= min(SLOTS, c):
                    break
                self.state["jobs"].remove(job)
                job.update(remote=scorer, started=time.time())
                mine.append(job)
                spec = {"job": {k: v for k, v in job.items() if k not in ("remote", "started")},
                        "wall": self.job_wall(job), "token_cap": self.job_token_cap(job), "concurrency": c}
                attempt = RemoteAttempt(self, job, scorer, spec)
                self.running[job["id"]] = (job, asyncio.create_task(self.play(job, attempt)))
                self.event_log({"event": "rsi_game_dispatched", "id": job["id"], "scorer": scorer,
                                "kind": job["kind"], "candidate": job["candidate"], "game": job["game"]})

    def loop_hours(self):
        return self.state["meters"]["loop_s"] / 3600

    # ----- candidates --------------------------------------------------------------------------------------------
    def live_candidate(self):
        return self.state["stack"]["live"]

    def concurrency(self):
        """Sessions on the GPU now (play slots + the proposer): the concurrency of the candidates whose games are
        running; it changes only between drained batches (``launch``)."""
        current = self.state.get("concurrency")
        if current is None:
            live = self.state["candidates"].get(self.live_candidate() or "", {})
            current = self.state["concurrency"] = int(live.get("concurrency", 14))
        return current

    def slots(self):
        return max(1, min(SLOTS, self.concurrency() - 1))

    def job_concurrency(self, job):
        """A game runs at its own candidate's concurrency (its time slice and its GPU share); smoke games are short
        checks and run at whatever is running."""
        if job["kind"] == "smoke":
            return self.concurrency()
        return max(2, min(SLOTS + 1, int(self.state["candidates"][job["candidate"]].get("concurrency", 14))))

    def job_wall(self, job):
        if job["kind"].startswith("heldout"):
            return float(self.cfg["heldout_wall_seconds"])
        if job["kind"] == "smoke":
            return float(self.cfg["smoke_seconds"])
        own = self.state["candidates"].get(job["candidate"], {}).get("game_seconds")
        full = float(own) if own else self.job_concurrency(job) * float(self.cfg["slice_seconds"])
        if job["kind"] == "practice" and job.get("seconds"):  # the proposer's choice, up to the game wall
            return max(60.0, min(full, float(job["seconds"])))
        return full

    def job_token_cap(self, job):
        """Held-out: the format's cap (100K). Pool and practice: the candidate's own ``game_token_cap``
        (none by default). Smoke games: none."""
        if job["kind"].startswith("heldout"):
            return int(self.cfg["heldout_token_cap"])
        if job["kind"] in ("pool", "practice"):
            cap = self.state["candidates"].get(job["candidate"], {}).get("game_token_cap")
            return int(cap) if cap else None
        return None

    def candidate_dir(self, cid):
        return self.root / "archive" / cid / "tree"

    def session_meta(self):
        """The setup of the proposer session running now (a self-update takes effect only at the next session)."""
        session = getattr(getattr(self, "proposer", None), "session", None)
        return (session or {}).get("meta") or self.state["meta"]["live"]

    def register_candidate(self, cid, tree, parent, note, author, hold=False):
        problems = cand.check(tree)
        run = json.loads((tree / "player/run.json").read_text()) if not problems else {}
        item = {"parent": parent, "note": note[:2000], "author": author, "created": time.time(),
                "status": "rejected" if problems else "smoke", "problems": problems,
                "serving_hash": cand.serving_hash(tree), "concurrency": int(run.get("concurrency", 14)),
                "game_seconds": run.get("game_seconds"), "game_token_cap": run.get("game_token_cap"),
                "pool": {}, "pool_levels": {}, "hold": bool(hold),
                **({"meta": self.session_meta()} if author == "proposer" else {})}
        self.state["candidates"][cid] = item
        meta = {k: item[k] for k in ("parent", "note", "author", "created", "status", "problems", "serving_hash")}
        atomic_write(self.root / "archive" / cid / "meta.json", meta)
        if not problems and cid != "c0000":
            self.enqueue({"kind": "smoke", "candidate": cid, "game": self.smoke_game(), "seed": 0})
        elif not problems:
            item["status"] = "new"
        self.event_log({"event": "rsi_candidate", "candidate": cid, "status": item["status"], "problems": problems})
        if problems and cid != "c0000":  # refused by check(): nothing of it stays in the archive but its meta.json
            shutil.rmtree(tree, ignore_errors=True)
        self.mirror_candidate(cid)
        return item

    def smoke_game(self):
        games = self.state["practice_games"]
        return games[self.state["seq"] % len(games)]

    def mirror_candidate(self, cid):
        """Checkpoint copy of the candidate's own files. node_modules is not mirrored (a resume links gen 0's) and
        files over 512 MB are not either: they are listed in the candidate's ``not_checkpointed``, and after a
        relaunch such a candidate is ``unrestorable`` (not evaluated, not a parent, not held out)."""
        target = self.ckpt / "archive" / cid
        if (target / "tree").exists() or not (self.root / "archive" / cid / "tree").is_dir():
            return
        source = self.root / "archive" / cid
        left_out = []
        for root, dirs, files in os.walk(source):
            dirs[:] = [d for d in dirs if d != "node_modules"]
            rel = Path(root).relative_to(source)
            for name in files:
                path = Path(root) / name
                size = path.lstat().st_size
                if path.is_symlink() or size <= 512 * 1024 ** 2:
                    (target / rel).mkdir(parents=True, exist_ok=True)
                    shutil.copy2(path, target / rel / name, follow_symlinks=False)
                else:
                    left_out.append({"path": str(path.relative_to(source)), "bytes": size})
        self.keep_big(left_out, source, f"archive/{cid}")
        if cid in self.state["candidates"]:
            self.state["candidates"][cid]["not_checkpointed"] = left_out  # in big_dir, not in the checkpoint

    def view(self):
        """What the policy sees: pool scores, practice clear rates, lineage; never held-out."""
        live = self.state["stack"]["live_hash"]
        cands = {}
        for cid, item in self.state["candidates"].items():
            if item["status"] in ("rejected", "smoke", "unrestorable", "retired"):
                continue
            cands[cid] = {"status": item["status"], "parent": item.get("parent"), "pool": item["pool"],
                          "pool_levels": item.get("pool_levels", {}), "pool_runs": item.get("pool_runs", {}),
                          "live": item["serving_hash"] == live, "stack_reverted": self.stack_reverted(cid),
                          # fixed-game jobs do not count: the policy still queues its own games beside them
                          "queued": sum(1 for j in self.state["jobs"] if j["candidate"] == cid and j["kind"] == "pool"
                                        and not j.get("fixed") and not j.get("anchor"))
                          + sum(1 for j, _ in self.running.values() if j["candidate"] == cid and j["kind"] == "pool"
                                and not j.get("fixed") and not j.get("anchor"))}
        view = {"candidates": cands, "pool_games": self.state["pool_games"], "base_pool_games": list(self.base_pool),
                "fixed_pool_games": list(self.fixed_pool),
                "practice": self.state["practice"], "practice_games": self.state["practice_games"],
                "seed": self.state["seq"]}
        if self.cfg["paired_minibatch"]:  # owner switch: gen 0's selection draws paired minibatches
            view["policy"] = {"paired": True}
        return view

    def start_fixed(self):
        """Bundle pool games new to a resumed state join the pool; the first leg with fixed games re-scores gen 0
        and the best of that moment (by the earlier rule) on them."""
        for role in ("pool", "practice"):  # bundle games new to a resumed state (13 Oct: the VGDL practice games)
            known = set(self.state[f"{role}_games"])
            self.state[f"{role}_games"] += [g for g in sorted(self.points.get(role, {})) if g not in known]
        if self.fixed_pool and "fixed_since" not in self.state:
            best = self.best(require_fixed=False) or "c0000"
            self.state.update(fixed_since=time.time(), fixed_rescore=sorted({"c0000", best}))
            self.event_log({"event": "rsi_fixed_pool", "games": self.fixed_pool, "rescore": self.state["fixed_rescore"]})

    def ensure_fixed(self):
        """Every candidate evaluated since the fixed games came in (and gen 0 and the best of that moment) plays all
        of them: queued here, never cancelled by the policy; a game that fails for good three times is given up."""
        if not self.fixed_pool:
            return
        rescore = set(self.state.get("fixed_rescore") or [])
        for cid, item in self.state["candidates"].items():
            if item["status"] in ("rejected", "failed", "unrestorable", "smoke", "held", "retired"):
                continue
            if cid not in rescore and not (item["status"] == "new" or item.get("fixed_queued")):
                continue
            item["fixed_queued"] = True
            tries = item.setdefault("fixed_tries", {})
            for game in self.fixed_pool:
                if game in item["pool"] or self.pending("pool", cid, game) or tries.get(game, 0) >= 3:
                    continue
                tries[game] = tries.get(game, 0) + 1
                self.enqueue({"kind": "pool", "candidate": cid, "game": game, "seed": 0, "fixed": True})
            if item["status"] == "done" and self.pool_pending(cid):  # a new one stays new for the policy's minibatch
                item["status"] = "evaluating"

    async def run_policy(self):
        """One policy decision at a time (games finishing together would otherwise queue the same evaluation twice);
        a (candidate, game) pool job already queued or running is not queued again."""
        if self.cfg["role"] == "heldout":
            return  # the held-out machine plays its requests only
        if not hasattr(self, "policy_lock"):
            self.policy_lock = asyncio.Lock()
        async with self.policy_lock:
            self.ensure_fixed()
            await self._run_policy()

    def pool_pending(self, cid):
        return any(j["candidate"] == cid and j["kind"] == "pool"
                   for j in [*self.state["jobs"], *(job for job, _ in self.running.values())])

    def pending(self, kind, cid, game):
        return any(j["kind"] == kind and j["candidate"] == cid and j["game"] == game
                   for j in [*self.state["jobs"], *(job for job, _ in self.running.values())])

    def check_decision(self, decision, view):
        """A policy decision with every field checked (a malformed one raises ValueError: the loop then uses gen 0's
        policy). Returns the cleaned decision."""
        if not isinstance(decision, dict):
            raise ValueError("decision: a JSON object")
        cands = view["candidates"]
        known = set(view["pool_games"])
        practice_known = set(view["practice_games"])

        def ident(value, what):
            if not isinstance(value, str) or value not in cands:
                raise ValueError(f"{what}: {value!r} is not a candidate in the view")
            return value

        def items(key, games_known, cap):
            out = []
            value = decision.get(key, [])
            if not isinstance(value, list):
                raise ValueError(f"{key}: a list")
            for item in value[:1000]:
                if not isinstance(item, dict) or not isinstance(item.get("games", []), list):
                    raise ValueError(f"{key}: items are {{'candidate': id, 'games': [...]}}")
                games = item.get("games", [])
                bad = [g for g in games if not isinstance(g, str) or g not in games_known]
                if bad:
                    raise ValueError(f"{key}: unknown games {bad[:3]!r}")
                out.append({"candidate": ident(item.get("candidate"), f"{key}.candidate"),
                            "games": list(dict.fromkeys(games))[:cap]})
            return out

        parent = decision.get("parent")
        cancel = decision.get("cancel", [])
        if not isinstance(cancel, list):
            raise ValueError("cancel: a list of candidate ids")
        best = decision.get("best")
        return {"parent": ident(parent, "parent") if parent is not None else None,
                "evaluate": items("evaluate", known, 200), "practice": items("practice", practice_known, 20),
                "cancel": [ident(c, "cancel") for c in cancel[:1000]],
                "best": ident(best, "best") if best is not None else None}

    async def _run_policy(self):
        view = self.view()
        decision, error = await self.proposer.policy(view)
        if decision is not None:
            try:
                decision = self.check_decision(decision, view)
            except ValueError as exc:
                decision, error = None, f"invalid decision: {exc}"
        if decision is None:
            fallback = self.state.setdefault("policy_fallback", {"count": 0})
            fallback.update(count=fallback["count"] + 1, last=time.time(), error=(error or "")[:300],
                            meta=self.state["meta"]["live"])
            if error and (error != fallback.get("told") or time.time() - fallback.get("told_at", 0) > 600):
                fallback.update(told=error, told_at=time.time())
                self.event_log({"event": "rsi_policy_fallback", "error": error[:300]})
                self.proposer.notify({"event": "policy_fallback", "meta": self.state["meta"]["live"],
                                      "error": error[:500], "note": "gen 0's policy decided instead"})
            try:
                decision = self.check_decision(gen0_policy.decide(view), view)
            except Exception as exc:  # noqa: BLE001 - never stop the loop over a decision
                self.event_log({"event": "rsi_policy_failed", "error": f"{type(exc).__name__}: {exc}"[:300]})
                return
        self.apply_decision(decision, view)

    def apply_decision(self, decision, view):
        self.state["suggested_parent"] = decision["parent"]
        self.state["policy_best"] = decision["best"]  # the policy's own pick for held-out; ``best()`` checks it
        for cid in decision["cancel"]:  # early stopping: drop the candidate's queued pool games
            before = len(self.state["jobs"])
            self.state["jobs"] = [j for j in self.state["jobs"]
                                  if not (j["kind"] == "pool" and j["candidate"] == cid
                                          and not j.get("fixed") and not j.get("anchor"))]
            item = self.state["candidates"][cid]
            if before != len(self.state["jobs"]) and item["status"] == "evaluating" and not any(
                    j["candidate"] == cid and j["kind"] == "pool" for j, _ in self.running.values()):
                item["status"] = "done"
        for item in decision["evaluate"]:
            cid = item["candidate"]
            if not view["candidates"][cid]["live"]:
                continue
            for game in item["games"]:
                if not self.pending("pool", cid, game):
                    self.enqueue({"kind": "pool", "candidate": cid, "game": game, "seed": 0})
            if self.state["candidates"][cid]["status"] == "new" and self.pool_pending(cid):
                self.state["candidates"][cid]["status"] = "evaluating"
        for item in decision["practice"]:
            for game in item["games"]:
                if game in self.points.get("practice", {}) and not self.pending("practice", item["candidate"], game):
                    self.enqueue({"kind": "practice", "candidate": item["candidate"], "game": game, "seed": 0})

    # ----- jobs --------------------------------------------------------------------------------------------------
    def enqueue(self, job):
        job = {"id": self.new_id("j"), **job, "queued": time.time()}
        self.state["jobs"].append(job)
        return job

    def trusted_live(self):
        return self.state["stack"]["live_hash"] == self.state["candidates"]["c0000"]["serving_hash"]

    def heldout_exclusive(self):
        """Held-out games always play alone (gen 0's and the best's alike, so both are measured the same way): no
        other game and no proposer session shares the server with them. On a stack that runs the proposer's server
        code the server also restarts from scratch afterwards (``scrub``), so it cannot carry what it saw to anything
        the proposer reads."""
        live = self.state["stack"]["live_hash"]
        return any(j["kind"].startswith("heldout") and self.state["candidates"][j["candidate"]]["serving_hash"] == live
                   for j in self.state["jobs"]) or any(j["kind"].startswith("heldout") for j, _ in
                                                       self.running.values())

    def next_job(self):
        """The first job by priority among those on the live stack whose wall fits before the deadline (not removed
        from the queue). In held-out exclusive mode only held-out games, after the others have drained."""
        live = self.state["stack"]["live_hash"]
        limit = self.deadline - self.cfg["end_margin_seconds"] - time.time()
        ready = [j for j in self.state["jobs"]
                 if self.state["candidates"].get(j["candidate"], {}).get("serving_hash") == live
                 and self.job_wall(j) <= limit]
        if self.heldout_exclusive():
            if any(not j["kind"].startswith("heldout") for j, _ in self.running.values()):
                return None
            if self.state.get("heldout_hold") != "held":
                return None  # the proposer's session is still finishing
            ready = [j for j in ready if j["kind"].startswith("heldout")]
        if self.state.get("budget_reached"):  # the approved loop GPU-h are spent: only a due held-out point plays
            ready = [j for j in ready if j["kind"].startswith("heldout")]
        return min(ready, key=lambda j: (PRIORITY[j["kind"]], j["queued"])) if ready else None

    def launch(self):
        if self.stop_launch or self.state.get("switching") or self.state.get("pending_switch"):
            return
        if time.time() + self.cfg["smoke_seconds"] + self.cfg["end_margin_seconds"] > self.deadline:
            self.stop_launch = True  # not even a smoke game fits any more: the run is ending
            self.event_log({"event": "rsi_stop_launching", "reason": "deadline"})
            return
        self.launch_local()
        self.dispatch_remote()

    def launch_local(self):
        while True:
            job = self.next_job()
            if job is None:
                return
            want = self.job_concurrency(job)
            if want != self.concurrency():
                if self.local_running():
                    return  # the running batch drains first; then the next batch runs at its own concurrency
                self.state["concurrency"] = want
                self.event_log({"event": "rsi_concurrency", "concurrency": want})
            cap = min(SLOTS, self.concurrency()) if job["kind"].startswith("heldout") else self.slots()
            if len(self.local_running()) >= cap:  # held-out plays alone (proposer paused): all C sessions its games
                return
            self.state["jobs"].remove(job)
            used = {j.get("slot") for j, _ in self.local_running()}
            job["slot"] = next(s for s in range(SLOTS) if s not in used)
            job["started"] = time.time()
            kind = job["kind"]
            set_name = {"heldout_a": "heldout_a", "heldout_b": "heldout_b"}.get(kind, "practice" if kind in (
                "practice", "smoke") else "pool")
            point = self.points[set_name][job["game"]]
            attempt = Attempt(self.runtime, job, point, self.candidate_dir(job["candidate"]), wall=self.job_wall(job),
                              meter="heldout" if kind.startswith("heldout") else "loop", games_set=set_name,
                              concurrency=self.concurrency(),
                              token_cap=self.job_token_cap(job))
            if kind.startswith("heldout"):
                self.state["heldout"]["accesses"].append({"at": time.time(), "job": job["id"], "game": job["game"],
                                                          "candidate": job["candidate"]})
            self.running[job["id"]] = (job, asyncio.create_task(self.play(job, attempt)))

    def requeue_running(self, reason):
        """Stop every running game and queue it again from the start (stack switch, server restart, deadline)."""
        for job, task in list(self.running.values()):
            self.state["jobs"].insert(0, {k: v for k, v in job.items() if k not in ("slot", "started", "remote")})
            self.running.pop(job["id"], None)
            task.cancel()
            self.event_log({"event": "rsi_game_requeued", "id": job["id"], "reason": reason})

    async def play(self, job, attempt):
        try:
            row = await attempt.run()
        except asyncio.CancelledError:
            return  # requeued by requeue_running
        except Exception as exc:  # noqa: BLE001 - the loop goes on; the job is retried once
            row = {"outcome": "error", "infrastructure": True, "error": f"{type(exc).__name__}: {exc}"[:500],
                   "traceback": traceback.format_exc()[-2000:]}
        self.running.pop(job["id"], None)
        await self.done(job, row)

    async def done(self, job, row):
        row = {**{k: job[k] for k in ("id", "kind", "candidate", "game", "seed")}, "started": job.get("started"),
               "ended": time.time(), **row}
        meter = "heldout" if job["kind"].startswith("heldout") else "loop"
        self.state["meters"][f"{meter}_tokens"] += int(row.get("tokens", 0))
        self.state["meters"][f"{meter}_actions"] += int(row.get("actions", 0))
        if row.get("infrastructure"):
            retries = job.get("retries", 0)
            if retries < 2 and not self.stop_launch:
                self.state["jobs"].insert(0, {**{k: v for k, v in job.items() if k not in ("slot", "started", "remote")},
                                              "retries": retries + 1})
            row["requeued"] = retries < 2
        if job["kind"].startswith("heldout"):
            # held-out results live in their own file (root-only); the state keeps that the game was played
            self.heldout_write({**row, "run": job.get("run")})
            row = {k: row.get(k) for k in ("id", "kind", "candidate", "game", "seed", "started", "ended", "outcome",
                                           "seconds", "tokens", "actions", "infrastructure", "requeued")}
        self.state["rows"].append(row)
        item = self.state["candidates"].get(job["candidate"])
        if job["kind"] == "smoke" and item is not None:
            ok = not row.get("infrastructure") and row.get("model_requests", 0) > 0 and row.get("actions", 0) > 0
            item["status"] = ("held" if item.get("hold") else "new") if ok else "rejected"
            if not ok:
                item["problems"] = [f"smoke game: {row.get('outcome')} {row.get('error', '')}"[:600],
                                    f"model requests {row.get('model_requests', 0)}, actions {row.get('actions', 0)}"]
        elif job["kind"] == "pool" and item is not None and not row.get("infrastructure"):
            if row.get("score") is not None:  # games the proposer added have no score (levels only)
                runs = item.setdefault("pool_runs", {}).setdefault(job["game"], [])
                runs.append(row["score"])
                item["pool"][job["game"]] = sum(runs) / len(runs)  # every play kept; the model sees their mean
            item["pool_levels"][job["game"]] = row["levels"]
            self.write_scores(job["candidate"])
        if job["kind"] == "pool" and item is not None and item["status"] == "evaluating" and \
                not self.pool_pending(job["candidate"]):
            item["status"] = "done"  # also when its last game failed for good on our side
        if job["kind"] == "practice" and not row.get("infrastructure"):
            self.state["practice"].setdefault(job["game"], []).append(
                {"levels": row["levels"], "total_levels": row["total_levels"], "candidate": job["candidate"]})
        elif job["kind"].startswith("heldout") and not row.get("infrastructure"):
            run = next(r for r in self.state["heldout"]["runs"] if r["id"] == job["run"])
            run["games"].setdefault(job["game"], []).append({"seed": job["seed"], "done": True})
        if job["kind"] in ("smoke", "pool", "practice") and not row.get("requeued"):
            self.probation_note(game=row)
        self.event_log({"event": "rsi_game", **{k: row.get(k) for k in (
            "id", "kind", "candidate", "game", "score", "levels", "outcome", "actions", "tokens", "seconds")}})
        if job["kind"] in ("smoke", "pool"):
            await self.run_policy()
        self.save("game")

    def heldout_write(self, row):
        """docs/rsi.md 4: nobody acts on held-out results. They go to <checkpoint>/heldout/results.jsonl only (root,
        0600); the reports at the registered points (``rsi_loop.report``) are the only place they are summarised."""
        folder = self.ckpt / "heldout"
        folder.mkdir(parents=True, exist_ok=True)
        os.chmod(folder, 0o700)
        path = folder / "results.jsonl"
        with path.open("a") as handle:
            handle.write(json.dumps(row) + "\n")
        os.chmod(path, 0o600)

    def heldout_results(self):
        path = self.ckpt / "heldout/results.jsonl"
        return [json.loads(line) for line in path.read_text().splitlines() if line.strip()] if path.exists() else []

    def write_scores(self, cid):
        item = self.state["candidates"][cid]
        atomic_write(self.root / "archive" / cid / "scores.json",
                     {"pool": item["pool"], "levels": item["pool_levels"], "status": item["status"]})

    # ----- held-out ----------------------------------------------------------------------------------------------
    def best(self, require_fixed=True):
        """The held-out candidate, on whatever stack it runs (``heldout_stack`` switches there and back), among
        candidates with at least ``best_min_games`` bundled-pool scores: the live policy's ``best`` if it names one
        of them, else gen 0's standing over the bundled pool games. A candidate whose stack failed probation is
        not eligible until that stack passes a probation again."""
        base = set(self.base_pool)
        scored = {}
        for c, v in self.state["candidates"].items():
            pool = {g: s for g, s in v["pool"].items() if g in base}
            if len(pool) >= self.cfg["best_min_games"] and v["status"] not in ("rejected", "failed", "unrestorable", "retired") \
                    and not self.stack_reverted(c) and (not require_fixed or set(self.fixed_pool) <= set(pool)):
                scored[c] = {**v, "pool": pool}
        pick = self.state.get("policy_best")
        if pick in scored:
            return pick
        stand = gen0_policy.standing(scored)
        ranked = sorted(stand, key=lambda c: -stand[c]["effect"])
        return ranked[0] if ranked else None

    def heldout_switch_failed(self, cid):
        """A held-out run whose stack failed to start twice is skipped (its games dropped, the run marked), so the
        loop does not retry it forever."""
        fails = self.state["heldout"].setdefault("switch_failures", {})
        fails[cid] = fails.get(cid, 0) + 1
        if fails[cid] < 2:
            return
        fails[cid] = 0
        self.state["jobs"] = [j for j in self.state["jobs"]
                              if not (j["kind"].startswith("heldout") and j["candidate"] == cid)]
        for run in self.state["heldout"]["runs"]:
            if run["open"] and run["candidate"] == cid:
                run["skipped"] = "its stack did not start (twice)"
        self.event_log({"event": "rsi_heldout_skipped", "candidate": cid})

    def heldout_due(self):
        if self.cfg["role"] == "heldout":
            return
        h = self.state["heldout"]
        hours = self.loop_hours()
        if h["next_a"] is None:
            h["next_a"], h["next_ab"] = self.cfg["heldout_a_every"], self.cfg["heldout_ab_every"] or 0
        if any(r["open"] for r in h["runs"]):
            return
        fmt = self.cfg["heldout_format"]
        runs = [r for r in h["runs"] if r.get("format") == fmt and not r.get("skipped")]
        if hours >= self.cfg["heldout_a_every"] and not any(
                r["candidate"] == "c0000" and "heldout_a" in r["sets"] for r in runs):
            self.open_heldout("c0000", ("heldout_a"), 1)  # gen 0's first point in this format
            return
        best = self.best()
        if self.cfg["heldout_ab_every"] and hours >= h["next_ab"]:
            mark = h["next_ab"]
            h["next_ab"] += self.cfg["heldout_ab_every"]
            h["next_a"] = max(h["next_a"], h["next_ab"] - self.cfg["heldout_ab_every"] + self.cfg["heldout_a_every"])
            sets, seeds = ("heldout_a", "heldout_b"), self.cfg["heldout_seeds"]
            frozen = next((r for r in runs if r["candidate"] == "c0000" and r["seeds"] == seeds
                           and set(r["sets"]) == set(sets)), None) if self.cfg["gen0_once"] else None
            ids = [frozen["id"]] if frozen else [self.open_heldout("c0000", sets, seeds)["id"]]
            if best and best != "c0000":
                ids.append(self.open_heldout(best, sets, seeds)["id"])
            totals = {kind: {g: p.get("total_levels") for g, p in self.points.get(kind, {}).items()} for kind in sets}
            h.setdefault("points", []).append({"loop_hours": mark, "runs": ids, "best": best, "totals": totals,
                                               "report": None})
            self.event_log({"event": "rsi_heldout_point", "loop_hours": mark, "runs": ids, "best": best})
        elif hours >= h["next_a"] and self.state["candidates"]["c0000"].get("status") != "evaluating":
            h["next_a"] += self.cfg["heldout_a_every"]  # (after a gen 0 re-anchor: once gen 0 is re-scored)
            if best and best != h["last_a_best"]:
                for cid in dict.fromkeys(["c0000", best] if not any(r["candidate"] == "c0000" for r in runs)
                                         else [best]):
                    self.open_heldout(cid, ("heldout_a",), 1)
                h["last_a_best"] = best

    def open_heldout(self, cid, sets, seeds):
        run = {"id": self.new_id("h"), "candidate": cid, "sets": list(sets), "seeds": seeds, "games": {},
               "open": True, "loop_hours": round(self.loop_hours(), 3), "opened": time.time(),
               "format": self.cfg["heldout_format"]}
        self.state["heldout"]["runs"].append(run)
        if self.cfg["heldout_mode"] == "remote" and self.cfg["role"] == "loop":
            run.update(open=False, remote=True, requested=run["opened"])
            request = {k: run[k] for k in ("id", "candidate", "sets", "seeds", "format", "loop_hours", "requested")}
            request["serving_hash"] = self.state["candidates"][cid]["serving_hash"]
            atomic_write(self.ckpt / "heldout/requests" / f"{run['id']}.json", request)
            self.event_log({"event": "rsi_heldout_request", "run": run["id"], "candidate": cid, "sets": sets,
                            "seeds": seeds})
            return run
        for kind in sets:
            for seed in range(seeds):
                for game in sorted(self.points.get(kind, {})):
                    self.enqueue({"kind": kind, "candidate": cid, "game": game, "seed": seed, "run": run["id"]})
        self.event_log({"event": "rsi_heldout_open", "run": run["id"], "candidate": cid, "sets": sets, "seeds": seeds})
        return run

    def close_heldout(self):
        for run in self.state["heldout"]["runs"]:
            if run["open"] and not any(j.get("run") == run["id"] for j in self.state["jobs"]) and not any(
                    j.get("run") == run["id"] for j, _ in self.running.values()):
                run["open"] = False
                run["closed"] = time.time()
                self.event_log({"event": "rsi_heldout_closed", "run": run["id"]})  # numbers stay in their file
        for point in self.state["heldout"].get("points", []):
            runs = [r for r in self.state["heldout"]["runs"] if r["id"] in point["runs"]]
            if point.get("report") is None and runs and not any(r["open"] for r in runs):
                if self.cfg["role"] == "loop" and any(r.get("remote") for r in runs):
                    point["report"] = "remote"  # the held-out machine's results (its own output) make it
                    continue
                from .report import point_report
                folder = self.ckpt / "reports"
                folder.mkdir(parents=True, exist_ok=True)
                path = folder / f"point-{point['loop_hours']:g}h.json"
                report = point_report(point["loop_hours"], runs, self.heldout_results(), point["totals"])
                report["skipped_runs"] = {r["id"]: r["skipped"] for r in runs if r.get("skipped")}
                path.write_text(json.dumps(report, indent=1) + "\n")
                point["report"] = str(path.relative_to(self.ckpt))
                self.event_log({"event": "rsi_report_written", "loop_hours": point["loop_hours"],
                                "report": point["report"]})  # the numbers stay in the file
        back = self.state["stack"].get("return_to")
        if self.cfg["role"] == "heldout":
            return  # the held-out machine ends after its runs: no way back to switch to
        if back and not any(r["open"] for r in self.state["heldout"]["runs"]) and not self.state.get("pending_switch"):
            self.state["stack"]["return_to"] = None
            if self.state["candidates"][back]["serving_hash"] != self.state["stack"]["live_hash"]:
                self.request_switch(back, None, purpose="return")

    def heldout_stack(self):
        """A held-out run of a candidate on another stack (gen 0 after the proposer moved the stack, a best on an
        older stack) switches to that stack and back afterwards; the switches count as held-out time."""
        if self.state.get("pending_switch") or self.state.get("switching"):
            return
        live = self.state["stack"]["live_hash"]
        waiting = [j for j in self.state["jobs"] if j["kind"].startswith("heldout")]
        if not waiting or any(j["kind"].startswith("heldout") for j, _ in self.running.values()):
            return
        if any(self.state["candidates"][j["candidate"]]["serving_hash"] == live for j in waiting):
            return
        target = min(waiting, key=lambda j: j["queued"])["candidate"]
        if self.state["stack"].get("return_to") is None:
            self.state["stack"]["return_to"] = self.state["stack"]["live"]
        self.request_switch(target, None, purpose="heldout")

    # ----- stack switches ----------------------------------------------------------------------------------------
    def request_switch(self, cid, gpu_job, purpose="proposer"):
        self.state["pending_switch"] = {"candidate": cid, "gpu_job": gpu_job, "requested": time.time(),
                                        "purpose": purpose}

    async def switch_stack(self, new, old, gpu_job=None, new_tag=None):
        """``Stacks.switch`` with gen 0's trusted stack as the last way back; sets the live stack from the outcome.
        When nothing comes back, the loop stops launching and the leg ends (a relaunch boots gen 0)."""
        fallback = (self.candidate_dir("c0000") / "serving", "c0000")
        try:
            record = await self.stacks.switch(self.candidate_dir(new) / "serving", new_tag or new,
                                              self.candidate_dir(old) / "serving", old, gpu_job=gpu_job,
                                              fallback=fallback)
        except Exception as exc:  # noqa: BLE001
            record = {"from": old, "to": new, "ok": False, "live": None,
                      "error": clip(f"{type(exc).__name__}: {exc}")}
            self.event_log({"event": "rsi_no_stack", **record})
            self.stop_launch = True
            return record
        live = {"new": new, "old": old, "fallback": "c0000"}[record["live"]]
        self.state["stack"]["live"] = live
        self.state["stack"]["live_hash"] = self.state["candidates"][live]["serving_hash"]
        await self.set_gate(live)
        return record

    async def set_gate(self, cid):
        """The proxy's KV admission gate for the live stack: on when its ``serving/study.json`` has ``rsi_gate``
        (``{"margin": 0.5..1.0}``), sized from the running server's own KV capacity; gen 0 has none."""
        from .candidates import read_inside
        from .feedback import server_metrics
        from .proxy import Gate

        self.proxy.gate, gate = None, None
        serving = self.candidate_dir(cid) / "serving"
        try:
            if (serving / "study.json").exists():
                gate = json.loads(read_inside(serving, "study.json")).get("rsi_gate")
        except (OSError, ValueError):
            gate = None
        if not isinstance(gate, dict):
            return
        margin = gate.get("margin", 0.9)
        if type(margin) not in (int, float) or not 0.5 <= margin <= 1.0:
            self.event_log({"event": "rsi_gate", "candidate": cid, "on": False, "why": "margin must be 0.5..1.0"})
            return
        try:
            async with self.proxy.client.get(self.proxy.upstream + "/metrics", timeout=10) as response:
                capacity = server_metrics(await response.text()).get("kv_capacity_tokens")
        except Exception:  # noqa: BLE001
            capacity = None
        if not capacity:
            self.event_log({"event": "rsi_gate", "candidate": cid, "on": False, "why": "no KV capacity on /metrics"})
            return
        self.proxy.gate = Gate(int(capacity * margin))
        self.event_log({"event": "rsi_gate", "candidate": cid, "on": True, "capacity_tokens": self.proxy.gate.capacity})

    def stack_reverted(self, cid):
        """The candidate's serving stack failed a probation and has not passed one since (gen 0's never counts)."""
        h = self.state["candidates"][cid]["serving_hash"]
        return h != self.state["candidates"]["c0000"]["serving_hash"] and h in self.state["stack"].get("reverted", {})

    def mark_reverted(self, watch):
        """A stack that failed probation: its candidates leave best() and held-out (queued held-out games dropped,
        open runs marked skipped) until the same stack passes a probation."""
        h = self.state["candidates"][watch["stack"]]["serving_hash"]
        if h == self.state["candidates"]["c0000"]["serving_hash"]:
            return
        self.state["stack"].setdefault("reverted", {})[h] = {
            "candidate": watch["stack"], "at": time.time(), "games": watch["games"], "failed_games": watch["bad"],
            "sessions_without_model": watch["bad_sessions"]}
        hit = {c for c in self.state["candidates"] if self.stack_reverted(c)}
        self.state["jobs"] = [j for j in self.state["jobs"]
                              if not (j["kind"].startswith("heldout") and j["candidate"] in hit)]
        for run in self.state["heldout"]["runs"]:
            if run["open"] and run["candidate"] in hit:
                run["skipped"] = "its stack failed probation"
        if self.state["heldout"].get("last_a_best") in hit:
            self.state["heldout"]["last_a_best"] = None
        self.event_log({"event": "rsi_stack_ineligible", "stack": watch["stack"], "candidates": sorted(hit)})

    def probation_note(self, game=None, session=None):
        """A stack the proposer adopted is watched: if half of its first 6+ loop games fail on our side or never reach
        the model, or two proposer sessions in a row make no model request, the loop switches back to the stack
        before it (``revert``). 20 games without that end the watch. "Reach the model": completion tokens > 0."""
        watch = self.state["stack"].get("probation")
        if not watch or watch["stack"] != self.state["stack"]["live"] or self.state.get("pending_switch"):
            return
        if game is not None:
            watch["games"] += 1
            watch["bad"] += int(bool(game.get("infrastructure")) or not game.get("tokens"))
        if session is not None:
            watch["sessions"] += 1
            watch["bad_sessions"] = watch["bad_sessions"] + 1 if not session.get("tokens") else 0
        crippled = (watch["games"] >= 6 and watch["bad"] * 2 >= watch["games"]) or watch["bad_sessions"] >= 2
        if crippled:
            self.state["stack"]["probation"] = None
            self.event_log({"event": "rsi_stack_revert", **watch})
            self.mark_reverted(watch)
            self.proposer.notify({"event": "stack_reverted", "stack": watch["stack"], "back_to": watch["back_to"],
                                  "games": watch["games"], "failed_games": watch["bad"],
                                  "sessions_without_model": watch["bad_sessions"]})
            self.request_switch(watch["back_to"], None, purpose="revert")
        elif watch["games"] >= 20:
            self.state["stack"]["probation"] = None
            passed = self.state["stack"].get("reverted", {}).pop(
                self.state["candidates"][watch["stack"]]["serving_hash"], None)
            if passed:
                self.event_log({"event": "rsi_stack_eligible", "stack": watch["stack"]})

    async def maybe_switch(self):
        pending = self.state.get("pending_switch")
        if not pending:
            return
        waited = time.time() - pending["requested"]
        if self.running and waited < self.cfg["drain_seconds"]:
            return
        self.requeue_running("stack switch")
        await asyncio.sleep(1)
        self.tick_meters()  # time so far by the ordinary split; the switch itself by its purpose (below)
        self.state["switching"] = True
        await self.proposer.pause()
        old = self.state["stack"]["live"]
        new = pending["candidate"]
        gpu_job = None
        if pending.get("gpu_job"):
            gpu_job = lambda: self.proposer.gpu_job(pending["gpu_job"])  # noqa: E731
        record = await self.switch_stack(new, old, gpu_job=gpu_job)
        purpose = pending.get("purpose")
        self.state["stack"]["history"].append({**record, "purpose": purpose})
        if purpose == "heldout" and not record.get("ok"):
            self.heldout_switch_failed(new)
        if purpose == "proposer" and record.get("ok"):  # watched until it has shown it can serve the loop
            self.state["stack"]["probation"] = {"stack": new, "back_to": old, "since": time.time(), "games": 0,
                                                "bad": 0, "sessions": 0, "bad_sessions": 0}
        elif purpose in ("proposer", "revert"):
            self.state["stack"]["probation"] = None
        self.tick_meters()  # still switching, purpose known: held-out switches go to the held-out meter
        self.state["pending_switch"] = None
        self.state["switching"] = False
        if self.state.get("heldout_hold"):  # any switch restarts the server: the one that saw held-out games is gone
            self.state["heldout_hold"] = None
        if not self.state.get("heldout_hold"):
            self.proposer.resume()
        if pending.get("purpose") == "proposer":
            self.proposer.notify({"event": "stack_switch", **{k: record.get(k) for k in (
                "to", "ok", "error", "seconds", "smoke")}})
        self.event_log({"event": "rsi_stack_switch", **record})
        await self.run_policy()
        self.save("switch")

    # ----- games the proposer writes -----------------------------------------------------------------------------
    async def runtime_add_game(self, host, path, role):
        """Vet a game the proposer wrote (sandboxed as rsigame, no network): ``<path>/environment_files/<prefix>/<ver>/``
        and ``<path>/solution.json`` (per level, a list of actions like "ACTION1" or "ACTION6@x,y"). It must replay its
        solution twice to a WIN with identical boards, and random play must not be trivial (``pool_rule``)."""
        if role not in ("practice", "pool"):
            raise ValueError("role: practice or pool")
        rel = host.work_rel(path)
        # Copied through descriptors (no link followed; a link at the top refused), then checked and read only on the
        # root-owned copy: the proposer cannot change what is vetted after the checks.
        staging = self.root / "root/vet" / secrets.token_hex(6)
        try:
            await asyncio.to_thread(cand.copy_untrusted, host.ws, rel, staging)
        except (OSError, ValueError) as exc:
            shutil.rmtree(staging, ignore_errors=True)
            raise ValueError(f"path: a plain folder under /workspace/work ({type(exc).__name__})") from None
        odd = [str(Path(r, n).relative_to(staging)) for r, ds, fs in os.walk(staging) for n in ds + fs
               if Path(r, n).is_symlink() or not (Path(r, n).is_dir() or Path(r, n).is_file())]
        if odd:
            shutil.rmtree(staging, ignore_errors=True)
            raise ValueError(f"only plain files and folders in a game folder: {odd[:5]}")
        subprocess.run(["chown", "-R", "root:root", str(staging)], check=True)
        subprocess.run(["chmod", "-R", "a+rX,a-w", str(staging)], check=True)
        metas = list((staging / "environment_files").glob("*/*/metadata.json"))
        if len(metas) != 1 or not (staging / "solution.json").is_file():
            shutil.rmtree(staging, ignore_errors=True)
            raise ValueError("exactly one game under environment_files/<prefix>/<version>/, and solution.json")
        game_id = str(json.loads(metas[0].read_text())["game_id"])
        # pool and practice only: the answer must not tell whether a held-out game has that prefix (held-out games
        # live in their own folders, so a shared prefix cannot collide)
        if any(game_id[:4] == g[:4] for role_ in ("pool", "practice") for g in self.points.get(role_, {})):
            shutil.rmtree(staging, ignore_errors=True)
            raise ValueError(f"{game_id}: the 4-character prefix is taken")
        from .attempt import RUNTIME
        argv = sandbox.command(
            [str(RUNTIME / "broker/bin/python"), "-I", str(RUNTIME / "gamelib/vet.py"),
             "/vet/game/environment_files", game_id, "/vet/game/solution.json"],
            sandbox.VET, None, {"HOME": "/tmp", "PATH": "/usr/bin:/bin"}, cwd="/tmp", memory=self.runtime.memory("vet"),
            extra_binds=[(staging, "/vet/game", True)], hide=self.runtime.hidden(),
            tmp_size="4g", shm_size="256m", net="none")
        process = await asyncio.create_subprocess_exec(*argv, stdout=asyncio.subprocess.PIPE,
                                                       stderr=asyncio.subprocess.PIPE)
        try:
            out, err = await asyncio.wait_for(process.communicate(), timeout=900)
        finally:
            subprocess.run(["pkill", "-KILL", "-u", sandbox.VET], check=False)
        lines = [line for line in out.decode().splitlines() if line.startswith("{")]
        result = json.loads(lines[-1]) if lines else {"ok": False, "error": err.decode()[-800:]}
        trivial = registry.trivial(result, self.cfg["pool_rule"])
        ok = bool(result.get("ok") and result.get("final_state") == "WIN" and not trivial)
        self.event_log({"event": "rsi_game_added" if ok else "rsi_game_refused", "game": game_id, "role": role,
                        "vet": {k: result.get(k) for k in ("ok", "problems", "deterministic", "final_state",
                                                            "random_levels", "error")}})
        if not ok:
            shutil.rmtree(staging, ignore_errors=True)
            problems = list(result.get("problems") or []) + (["random play clears it (trivial)"] if trivial else [])
            if result.get("final_state") != "WIN":
                problems.append("the solution does not reach WIN")
            return {"added": False, "vet": {"ok": False, "problems": problems[:10]}}
        folder = Path(metas[0]).parent.relative_to(staging / "environment_files")
        target = self.root / "games" / role / "environment_files" / folder
        if target.exists():  # left by an add that died before its save (the prefix check above passed)
            subprocess.run(["chmod", "-R", "u+w", str(target)], check=False)
            shutil.rmtree(target)
        shutil.copytree(staging / "environment_files" / folder, target)
        self.runtime.lock_games()
        solution = [lv["actions"] for lv in result["levels"]]
        # Pool games score like the bundled ones, with the proposer's own vetted solve as the reference action counts
        # (docs/rsi.md 3: added games count toward fitness); the held-out minimum counts bundled games only.
        point = {"id": f"{game_id}-game", "game": game_id, "split": role, "level_index": 0, "level": 1,
                 "through": True, "total_levels": len(solution), "baselines": solution if role == "pool" else None,
                 "proposer_solution": solution, "author": "proposer"}
        self.points.setdefault(role, {})[game_id] = point
        shutil.copytree(target, self.ckpt / "games" / role / "environment_files" / folder, dirs_exist_ok=True)
        shutil.rmtree(staging, ignore_errors=True)
        self.state.setdefault("added_games", []).append({**point, "role": role, "added": time.time()})
        self.state[f"{role}_games"] = sorted(set(self.state[f"{role}_games"]) | {game_id})
        self.save("game added")
        return {"added": True, "game": game_id, "role": role, "levels": len(solution),
                "note": ("scored with your solution's action counts as the reference" if role == "pool"
                         else "practice: traces and clear rates")}

    # ----- main --------------------------------------------------------------------------------------------------
    async def setup(self):
        await asyncio.to_thread(self.runtime.prepare)
        await self.proxy.start()
        self.proposer = ProposerHost(self)
        if not self.state["candidates"]:
            tree = self.candidate_dir("c0000")
            await asyncio.to_thread(self.runtime.install_gen0, tree)
            self.state["stack"]["live"] = "c0000"
            self.register_candidate("c0000", tree, None, "gen 0: stock Prime Agent ARC setup", "build")
            self.state["stack"]["live_hash"] = self.state["candidates"]["c0000"]["serving_hash"]
        else:
            await asyncio.to_thread(self.runtime.restore, self.ckpt, self.source / self.cfg["resume"], self.state)
            resumed = self.source / self.cfg["resume"]
            for part in ("heldout", "reports"):
                if (resumed / part).is_dir() and not (self.ckpt / part).exists():
                    shutil.copytree(resumed / part, self.ckpt / part)
            if self.state.get("gen0_version", 1) < 2:
                await asyncio.to_thread(self.reanchor_gen0)
            if self.cfg.get("void_cut_since") and not self.state.get("cut_rows_voided"):
                self.void_cut_rows(float(self.cfg["void_cut_since"]))
            for cid in self.state["candidates"]:  # the proposer's archive view from the restored state
                if (self.root / "archive" / cid).is_dir():
                    self.write_scores(cid)
            for cid in self.state["candidates"]:  # this leg's checkpoint carries every restored tree (13 Oct: a
                if cid != "c0000":                 # second resume lost the trees the first had restored)
                    await asyncio.to_thread(self.mirror_candidate, cid)
        self.stacks.trusted_hash = self.state["candidates"]["c0000"]["serving_hash"]
        for run in self.state["heldout"]["runs"]:  # runs before the token-cap format
            if "format" not in run:
                run.update(format="time-slice", note="old held-out format (time slice): not comparable")
        self.start_fixed()
        live = self.state["stack"]["live"]
        if self.state["candidates"][live].get("status") == "unrestorable":
            live = None
        self.state["stack"]["live"] = "c0000"  # the bootstrap started gen 0's stack
        self.state["stack"]["live_hash"] = self.state["candidates"]["c0000"]["serving_hash"]
        if live and live != "c0000" and self.cfg["role"] == "loop":  # held-out machine: heldout_stack picks
            await self.switch_stack(live, "c0000")
        await self.proposer.setup()
        from .isolation import self_test
        await self_test(self)
        # the probes' own refused packets (expected): subtract from the frontier audit's egress counts
        self.state.setdefault("egress_from_isolation_probes", []).append(self.runtime.egress())
        if "unprivileged_serving" not in self.state:
            await self.probe_unprivileged_serving()
        self.operator_events()
        if self.cfg["role"] == "heldout":
            self.take_requests()
        else:
            await self.run_policy()
        self.save("start")

    def take_requests(self):
        """Held-out machine: the loop's state (restored from its checkpoint) is a fork; the loop's own queue is dropped
        and only the requested held-out runs play (all C slots, proposer never started)."""
        self.state["jobs"] = [j for j in self.state["jobs"] if j["kind"].startswith("heldout")]
        runs = self.state["heldout"]["runs"]
        for req in self.cfg["heldout_requests"]:
            run = next((r for r in runs if r["id"] == req["id"]), None)
            if run is None:
                run = {"id": req["id"], "games": {}, "loop_hours": req.get("loop_hours")}
                runs.append(run)
            if run.get("played_here"):
                continue  # a resumed held-out machine keeps what it played
            run.update(candidate=req["candidate"], sets=list(req["sets"]), seeds=int(req["seeds"]),
                       format=req.get("format", self.cfg["heldout_format"]), open=True, remote=False,
                       played_here=True, opened=time.time(), requested=req.get("requested"))
            have = {(j["kind"], j["game"], j["seed"]) for j in self.state["jobs"] if j.get("run") == run["id"]}
            for kind in run["sets"]:
                for seed in range(run["seeds"]):
                    for game in sorted(self.points.get(kind, {})):
                        if (kind, game, seed) not in have and not any(
                                p.get("seed") == seed for p in run["games"].get(game, [])):
                            self.enqueue({"kind": kind, "candidate": run["candidate"], "game": game, "seed": seed,
                                          "run": run["id"]})
            self.event_log({"event": "rsi_heldout_open", "run": run["id"], "candidate": run["candidate"],
                            "sets": run["sets"], "seeds": run["seeds"], "machine": "heldout"})
        asked = {req["id"] for req in self.cfg["heldout_requests"]}
        played = {row.get("run") for row in self.heldout_results()}  # earlier held-out machines' results, if merged in
        for point in self.state["heldout"].get("points", []):  # a point this machine completes is reported here
            if point.get("report") == "remote" and all(r in asked or r in played for r in point["runs"]):
                point["report"] = None

    def reanchor_gen0(self):
        """gen 0 is Prime's ARC setup with its autonomous limits (12 turns / 80K tokens / 30 min) lifted.
        The bundle's gen 0 tree (installed as c0000 on this machine) is that setup; the earlier gen 0 becomes c0000-v1
        (retired: never best, out of the policy view, its rows, practice records and held-out runs relabelled), and the
        new c0000 is queued on the fixed games (ensure_fixed) and on every other pool game a candidate has played
        (``anchor`` jobs: not cancellable), so it is paired with every candidate."""
        old_id, cands = "c0000-v1", self.state["candidates"]
        old = dict(cands["c0000"])
        cands[old_id] = {**old, "status": "retired", "note": "gen 0 v1: Prime's ARC setup with its default autonomous "
                         "limits (12 turns / 80K tokens / 30 min); retired 14 Oct (owner)", "retired": time.time()}
        tree = self.candidate_dir("c0000")
        target = self.candidate_dir(old_id)
        if not target.exists():
            from .candidates import copy_tree
            copy_tree(tree, target, link_big=True)
            practice = self.root / "archive/c0000/practice"
            if practice.is_dir():  # the old gen 0's practice traces are the retired one's
                shutil.move(str(practice), str(target.parent / "practice"))
            run_path = target / "player/run.json"
            run = json.loads(run_path.read_text())
            flags = ("--autonomous-max-turns", "--autonomous-max-continuations", "--autonomous-max-tokens",
                     "--autonomous-timeout-ms")
            argv, skip = [], False
            for a in run["argv"]:
                if skip:
                    skip = False
                    continue
                if a in flags:
                    skip = True
                    continue
                argv.append(a)
            run["argv"] = argv
            subprocess.run(["chmod", "u+w", str(run_path)], check=False)
            run_path.write_text(json.dumps(run, indent=1) + "\n")
            self.runtime.seal(target.parent)
        for row in self.state["rows"]:
            if row.get("candidate") == "c0000":
                row["candidate"] = old_id
        for plays in self.state["practice"].values():
            for play in plays:
                if play.get("candidate") == "c0000":
                    play["candidate"] = old_id
        for run in self.state["heldout"]["runs"]:
            if run["candidate"] == "c0000":
                run["candidate"] = old_id
        for access in self.state["heldout"]["accesses"]:
            if access.get("candidate") == "c0000":
                access["candidate"] = old_id
        self.state["heldout"]["last_a_best"] = None
        self.state["jobs"] = [j for j in self.state["jobs"] if j["candidate"] != "c0000"]
        cands["c0000"] = {**{k: v for k, v in old.items() if k not in ("fixed_tries", "pool_runs")},
                          "pool": {}, "pool_levels": {}, "pool_runs": {}, "status": "evaluating", "fixed_queued": True,
                          "note": "gen 0 (v2): Prime's ARC setup with its autonomous limits lifted",
                          "created": time.time(), "problems": []}
        played = sorted({g for c, v in cands.items() if c not in ("c0000", old_id) for g in v.get("pool", {})}
                        - set(self.fixed_pool))
        for game in played:
            self.enqueue({"kind": "pool", "candidate": "c0000", "game": game, "seed": 0, "anchor": True})
        self.state["gen0_version"] = 2
        self.event_log({"event": "rsi_gen0_reanchored", "retired": old_id, "anchor_games": len(played),
                        "fixed_games": len(self.fixed_pool)})

    def void_cut_rows(self, since):
        """14 Oct (operator bug, 83ec2392): from ``since`` on, a game whose session started a context compaction
        after agent_end was ended as "stopped" during that compaction, so it played only its first context. Those
        pool rows are voided (kept, marked), their scores taken out of the candidates' pool means, and the games
        queued again (fixed and gen 0 anchor games keep their flags)."""
        voided = 0
        for row in self.state["rows"]:
            if row.get("kind") != "pool" or row.get("outcome") != "stopped" or (row.get("started") or 0) < since \
                    or row.get("void"):
                continue
            row["void"] = "cut at its first context compaction (operator bug, 14 Oct); replayed"
            voided += 1
            item = self.state["candidates"].get(row["candidate"])
            if item is None:
                continue
            game = row["game"]
            runs = item.setdefault("pool_runs", {}).get(game) or []
            if row.get("score") is not None and row["score"] in runs:
                runs.remove(row["score"])
            if runs:
                item["pool"][game] = sum(runs) / len(runs)
            else:
                item["pool"].pop(game, None)
                item["pool_runs"].pop(game, None)
                item["pool_levels"].pop(game, None)
            job = {"kind": "pool", "candidate": row["candidate"], "game": game, "seed": row.get("seed", 0)}
            if game in self.fixed_pool:
                job["fixed"] = True
            if row["candidate"] == "c0000":
                job["anchor"] = True
            if not self.pending("pool", row["candidate"], game):
                self.state["jobs"].insert(0, {"id": self.new_id("j"), **job, "queued": time.time()})
            if item.get("status") == "done":
                item["status"] = "evaluating"
        for cid in self.state["candidates"]:
            if (self.root / "archive" / cid).is_dir():
                self.write_scores(cid)
        self.state["cut_rows_voided"] = {"at": time.time(), "since": since, "rows": voided}
        self.event_log({"event": "rsi_cut_rows_voided", "rows": voided, "since": since})

    async def probe_unprivileged_serving(self):
        """Once per run: gen 0's stack started the way a proposer-built stack is (rsiserve, its namespace) and
        smoke-tested; the outcome tells the proposer whether serving changes can be adopted at all. Gen 0's stack
        stays up either way (on failure the root restarts it)."""
        if self.state["stack"]["live"] != "c0000":
            return
        self.tick_meters()
        self.state["switching"] = True
        self.stacks.force_unprivileged = True
        try:
            record = await self.switch_stack("c0000", "c0000", new_tag="c0000-unprivileged-probe")
        finally:
            self.stacks.force_unprivileged = False
            self.tick_meters()
            self.state["switching"] = False
        self.state["unprivileged_serving"] = {k: record.get(k) for k in ("ok", "error", "seconds", "smoke")}
        self.event_log({"event": "rsi_unprivileged_serving_probe", **self.state["unprivileged_serving"]})

    def operator_events(self):
        """docs/rsi.md 7 (human-hours): operator actions on this run, from the bundle's ``configs/rsi_operator.jsonl``
        (one JSON object per action: id, at, by, what, human_seconds), plus this start; human_s sums the declared
        seconds."""
        log = self.state.setdefault("operator_log", [])
        seen = {e.get("id") for e in log}
        path = self.source / self.cfg["operator_log"]
        if path.is_file():
            for line in path.read_text().splitlines():
                if not line.strip():
                    continue
                entry = json.loads(line)
                if entry.get("id") in seen:
                    continue
                log.append({**entry, "ingested": time.time()})
                self.state["meters"]["human_s"] += float(entry.get("human_seconds") or 0)
        resumed = self.state.get("resumed") or []
        log.append({"id": f"start-{int(time.time())}", "at": time.time(), "by": "loop",
                    "what": f"resume from {resumed[-1]['from']}" if resumed else "fresh start", "human_seconds": 0})

    def stop_requested(self):
        return any(Path(p).exists() for p in (os.environ.get("ARENA_STOP_REQUEST", "/tmp/arena-stop-request"),
                                               str(self.output / "stop-request")))

    async def run(self):
        await self.setup()
        last_save = time.time()
        try:
            while True:
                self.tick_meters()
                if not self.stop_launch and shutil.disk_usage(self.root).free < 20 * 1024 ** 3:
                    self.stop_launch = True  # a full disk kills the run mid-write; end it cleanly and say why
                    self.event_log({"event": "rsi_stop_launching", "reason": "disk below 20 GiB free"})
                if self.cfg["role"] == "loop" and not self.state.get("budget_reached") and self.loop_hours() >= float(self.cfg["budget_loop_gpu_hours"]):
                    self.heldout_due()  # a held-out point due at the limit still runs (its own meter)
                    self.state["budget_reached"] = time.time()
                    self.event_log({"event": "rsi_budget_reached", "loop_hours": round(self.loop_hours(), 3),
                                    "approved": self.cfg["budget_loop_gpu_hours"]})
                if self.state.get("budget_reached") and not self.stop_launch and not self.heldout_exclusive() \
                        and not any(r["open"] for r in self.state["heldout"]["runs"]):
                    self.stop_launch = True
                    self.event_log({"event": "rsi_stop_launching", "reason": "approved loop GPU-h reached"})
                if self.stop_requested() and not self.stop_launch:
                    self.stop_launch = True
                    self.state.setdefault("operator_log", []).append(
                        {"id": f"stop-{int(time.time())}", "at": time.time(), "by": "operator",
                         "what": "stop request", "human_seconds": 0})
                    self.event_log({"event": "rsi_stop_launching", "reason": "stop request"})
                if self.stop_launch and not self.state.get("stop_at"):
                    self.state["stop_at"] = time.time()
                    if self.proposer.process is not None:
                        self.proposer.notify({"event": "leg_ending", "minutes_left_at_most":
                                              round(self.cfg["proposer_grace_seconds"] / 60),
                                              "note": "this session may finish; save notes and drafts under work/"})
                self.heldout_due()
                self.close_heldout()
                self.heldout_stack()
                await self.maybe_switch()
                await self.heldout_hold()
                await self.check_server()
                self.launch()
                grace_over = bool(self.state.get("stop_at")) and (
                    time.time() - self.state["stop_at"] > self.cfg["proposer_grace_seconds"])
                if self.cfg["role"] == "heldout":
                    if not self.running and not any(j["kind"].startswith("heldout") for j in self.state["jobs"]) \
                            and not self.state.get("switching") and not self.state.get("pending_switch"):
                        self.close_heldout()
                        self.event_log({"event": "rsi_heldout_machine_done"})
                        break
                else:
                    await self.proposer.tick(stop=self.stop_launch and (grace_over or self.proposer.process is None))
                self.state["frontier"] = {**self.proxy_frontier(), "egress": self.runtime.egress()}
                if self.stop_launch and not self.running and (self.proposer.process is None or grace_over):
                    break
                if time.time() > self.deadline - 120:
                    self.requeue_running("deadline")
                    break
                if time.time() - last_save > 60:
                    self.save("meters")
                    last_save = time.time()
                if time.time() - getattr(self, "last_control", 0) > 30:
                    self.write_control()
                    self.last_control = time.time()
                if time.time() - getattr(self, "last_mirror", 0) > 600:
                    await self.mirror_now()
                    self.last_mirror = time.time()
                await asyncio.sleep(5)
        finally:
            self.write_control(end=True)
            await self.proposer.stop()
            if not self.trusted_live():  # an unprivileged stack is stopped and wiped here, not by the bootstrap
                await asyncio.to_thread(self.stacks.stop)
            copied, work, other = self.mirror_all()
            self.state["big_files"] = {"work": work[:500], "not_checkpointed": other[:500]}
            self.save("end")
            await self.proxy.stop()
        return {"status": "complete", "loop_hours": round(self.loop_hours(), 3),
                "candidates": len(self.state["candidates"]), "games": len(self.state["rows"])}

    async def heldout_hold(self):
        """Held-out exclusive mode (``heldout_exclusive``): the proposer is paused while it lasts; afterwards the
        stack restarts from scratch (a ``scrub``, or the switch back) before the proposer resumes."""
        hold = self.state.get("heldout_hold")
        if self.heldout_exclusive():
            if not hold or hold == "waiting":
                if self.proposer.process is not None:  # its session may finish first (up to the grace)
                    if not hold:
                        self.state["heldout_hold"], self.state["hold_wait_at"] = "waiting", time.time()
                        self.proposer.notify({"event": "heldout_pause_coming", "minutes_left_at_most":
                                              round(self.cfg["proposer_grace_seconds"] / 60),
                                              "note": "this session may finish; save notes and drafts under work/"})
                    if time.time() - self.state["hold_wait_at"] < self.cfg["proposer_grace_seconds"]:
                        return
                self.state["heldout_hold"] = "held"
                await self.proposer.pause()
                self.event_log({"event": "rsi_heldout_exclusive", "live": self.live_candidate()})
            elif not self.proposer.paused:
                await self.proposer.pause()
            return
        if self.cfg["role"] == "heldout":
            return  # the machine ends after its runs; nothing the proposer reads runs on it
        if hold == "held" and not self.state.get("switching"):
            if self.trusted_live():  # gen 0's own server code: nothing to scrub
                self.state["heldout_hold"] = None
                self.proposer.resume()
                return
            if not self.state.get("pending_switch"):
                self.request_switch(self.live_candidate(), None, purpose="scrub")
            self.state["heldout_hold"] = "scrub"

    def proxy_frontier(self):
        """Non-Flash model calls refused by the proxy and calls with unknown keys, over every leg."""
        totals = self.proxy.totals
        now = {"non_flash": sum(t["non_flash"] for t in totals.values()),
               "unknown_key": totals["_unknown"]["unknown_key"] if "_unknown" in totals else 0}
        return {k: v + int(self.frontier_base.get(k) or 0) for k, v in now.items()}

    async def check_server(self):
        """Every 60 s: three failed health checks in a row restart the live stack (running games requeue)."""
        if self.state.get("switching") or time.time() - getattr(self, "last_health", 0) < 60:
            return
        self.last_health = time.time()
        import aiohttp
        try:
            async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=10)) as http:
                async with http.get(self.upstream.removesuffix("/v1") + "/health") as response:
                    healthy = response.status == 200
        except Exception:  # noqa: BLE001
            healthy = False
        self.unhealthy = 0 if healthy else self.unhealthy + 1
        if self.unhealthy < 3:
            return
        live = self.state["stack"]["live"]
        self.event_log({"event": "rsi_server_down", "live": live})
        self.requeue_running("server down")
        self.tick_meters()
        self.state["switching"] = True
        await self.proposer.pause()
        record = await self.switch_stack(live, "c0000", new_tag=f"{live}-restart")
        self.state["stack"]["history"].append({**record, "reason": "server down"})
        self.tick_meters()
        self.state["switching"] = False
        if self.state.get("heldout_hold") and not self.heldout_exclusive():
            self.state["heldout_hold"] = None  # restarted: the server that saw held-out games is gone
        if not self.state.get("heldout_hold"):
            self.proposer.resume()
        self.unhealthy = 0
        self.save("server restart")


async def run_loop(study, trial, **kwargs):
    loop = Loop(study, trial, **kwargs)
    if loop.cfg["role"] == "scorer":
        from .remote import Scorer
        return await Scorer(loop).run()
    return await loop.run()
