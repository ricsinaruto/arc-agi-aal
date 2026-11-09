"""Split pool evaluation across two VMs (rsi2_loop.exchange + Loop.split_pool_run), against a mock bucket.

The mock copies each side's ``rsi-exchange/out`` into the other side's ``exchange-in`` on a timer, like the Nebius
runner's sync, with optional faults: truncated files in transit, a dead worker, a slow (late) worker, a tampered
tree. Time is scaled: one real second is 1/SCALE game seconds, so a 4,095 s half takes ~0.4 s.
"""
import asyncio
import os
import json
import shutil
import time
from pathlib import Path

import pytest

from rsi2_loop import exchange as X
from rsi2_loop import loop as L

SCALE = 1e-4  # real seconds per game second
POOL = [f"g{i:02d}" for i in range(29)]
FRACTIONS = {"s1": 50 / 380, "s2": 150 / 380, "s3": 1.0}


def game_clock():
    t0 = time.time()
    return lambda: t0 + (time.time() - t0) / SCALE


def score(game, mark, delta=0.0):
    return {"score": (int(game[1:]) % 7) * 3 + {"s1": 1, "s2": 5, "s3": 9}[mark] + delta, "cont": 10.0}


def make_tree(root, extra=None):
    (root / "player/src/arena").mkdir(parents=True, exist_ok=True)
    (root / "player/study.json").write_text('{"a": 1}')
    (root / "player/src/arena/trial.py").write_text("x = 1\n")
    (root / "serving").mkdir(parents=True, exist_ok=True)
    (root / "serving/server.json").write_text("{}")
    for rel, text in (extra or {}).items():
        path = root / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text)
    return root


class Bucket:
    """Two sides' out folders, each copied into the other's exchange-in every ``period`` real seconds."""

    def __init__(self, tmp, period=0.004, truncate_first=False, lose=()):
        self.loop_out, self.worker_out = tmp / "L/rsi-exchange/out", tmp / "W/rsi-exchange/out"
        self.loop_in, self.worker_in = tmp / "L/exchange-in/w/rsi-exchange/out", tmp / "W/exchange-in/l/rsi-exchange/out"
        for p in (self.loop_out, self.worker_out, self.loop_in, self.worker_in):
            p.mkdir(parents=True, exist_ok=True)
        self.period, self.truncate_first, self.seen = period, truncate_first, set()
        self.blocked = set()  # "worker": its files stop reaching the loop (a dead VM)
        # like rclone --update: a file is copied when its time is newer than the last copy; a name in ``lose`` has its
        # first version skipped (an upload that raced a rewrite), so only a later time bump re-sends it
        self.lose, self.copied, self.lost = set(lose), {}, set()

    def copy(self, src, dst, side):
        if side in self.blocked:
            return
        for path in src.rglob("*"):
            if not path.is_file() or path.name.startswith("."):
                continue
            target = dst / path.relative_to(src)
            mtime = path.stat().st_mtime_ns
            if self.copied.get(str(path), -1) >= mtime:
                continue
            self.copied[str(path)] = mtime
            if path.name in self.lose and str(path) not in self.lost:
                self.lost.add(str(path))
                continue
            target.parent.mkdir(parents=True, exist_ok=True)
            data = path.read_bytes()
            key = (side, str(target), data)
            if self.truncate_first and key not in self.seen and path.suffix == ".json":
                self.seen.add(key)
                self.copied[str(path)] = -1  # completed at the next sync
                target.write_bytes(data[: len(data) // 2])  # a partial copy, completed at the next sync
                continue
            target.write_bytes(data)

    async def run(self, stop):
        while not stop.is_set():
            self.copy(self.loop_out, self.worker_in, "loop")
            self.copy(self.worker_out, self.loop_in, "worker")
            await asyncio.sleep(self.period)


class FakeRun:
    """Stands in for evaluate.Run: calls ``decide`` at each mark after the mark's (scaled) seconds."""

    made = []
    delta = 0.0
    slow = 1.0

    def __init__(self, host, cid, tree, *, kind, games, games_dir, wall, marks, tag, log_actions=False, **_):
        self.games, self.wall, self.marks, self.tag = list(games), wall, dict(marks), tag
        self.stopped = None
        self.actions = Path("/nonexistent/actions.jsonl")
        self.result = {}
        FakeRun.made.append(self)

    def stop_request(self, why):
        self.stopped = why

    async def go(self, decide):
        started = time.time()
        self.result["started"] = started
        last, out = 0, {}
        for mark, seconds in sorted(self.marks.items(), key=lambda kv: kv[1]):
            while (time.time() - started) < seconds * SCALE * self.slow:
                if self.stopped:
                    return {"status": "stopped", "started": started, "seconds": time.time() - started, "marks": out}
                await asyncio.sleep(0.002)
            verdict = decide(mark, {"games": {g: score(g, mark, self.delta) for g in self.games}, "t": seconds})
            out[mark] = {"verdict": verdict}
            last = seconds
            if verdict == "cut":
                return {"status": "complete", "started": started, "seconds": last, "exit": 0, "marks": out,
                        "cut": mark}
        return {"status": "complete", "started": started, "seconds": last, "exit": 0, "marks": out, "cut": None}


class Host:
    """The parts of Loop that split_pool_run uses."""

    split_pool_run = L.Loop.split_pool_run

    def __init__(self, tmp, bucket, gen0, now):
        self.pool = POOL
        self.cfg = {"mark_fractions": FRACTIONS, "slice_seconds": 273, "split_poll": 0.003}
        self.root = tmp / "root"
        self.dispatcher = X.Dispatcher(X.Mailbox(bucket.loop_out, bucket.loop_in), gen0, now=now)
        self.events, self.stop = [], False
        self.trees = {}

    def candidate_dir(self, cid):
        return self.trees[cid]

    def event_log(self, row):
        self.events.append(row)

    def save(self, reason=""):
        pass

    def stop_requested(self):
        return self.stop


def worker_job(now, slow=1.0, delta=0.0):
    """A worker's run_job: plays the half with FakeRun timing on the game clock, reporting each mark."""

    async def run_job(job, tree, report, cancelled):
        run = FakeRun(None, job["cid"], tree, kind="eval", games=job["games"], games_dir=None, wall=job["wall"],
                      marks=job["marks"], tag=job["jid"])
        run.slow, run.delta = slow, delta
        marks = {}
        report({"started": now(), "marks": {}})

        def decide(mark, scores):
            marks[mark] = scores
            report({"started": now(), "marks": marks})
            return "continue"

        async def watch():
            while True:
                await asyncio.sleep(0.003)
                if cancelled():
                    run.stop_request("cancelled")
                    return

        w = asyncio.create_task(watch())
        try:
            result = await run.go(decide)
        finally:
            w.cancel()
        return {"status": result["status"], "marks": marks, "cancelled": cancelled()}

    return run_job


async def run_worker(worker, stop):
    beats = asyncio.create_task(worker.heartbeat(stop))
    try:
        while not stop.is_set():
            if not await worker.step():
                await asyncio.sleep(0.003)
    finally:
        beats.cancel()


@pytest.fixture
def env(tmp_path, monkeypatch):
    monkeypatch.setattr(L, "Run", FakeRun)
    monkeypatch.setattr(X, "HEARTBEAT", 0.005)
    FakeRun.made, FakeRun.delta, FakeRun.slow = [], 0.0, 1.0
    gen0 = make_tree(tmp_path / "gen0")
    cand = make_tree(tmp_path / "cand", {"player/src/arena/trial.py": "x = 2\n", "player/src/arena/new.py": "y\n"})
    (cand / "player/study.json").unlink()  # a deletion travels in the manifest
    return tmp_path, gen0, cand


def decider(record, cut_at=None):
    def decide(mark, scores):
        record[mark] = scores
        return "cut" if mark == cut_at else ("accept" if mark == "s3" else "continue")
    return decide


async def scenario(tmp, gen0, cand, *, worker_kw=None, kill_worker_at=None, cut_at=None, bucket_kw=None,
                   worker_base=None, tamper=False, start_worker=True):
    now = game_clock()
    bucket = Bucket(tmp, **(bucket_kw or {}))
    host = Host(tmp, bucket, gen0, now)
    host.trees["c0001"] = cand
    worker = X.Worker(X.Mailbox(bucket.worker_out, bucket.worker_in), worker_base or gen0, tmp / "wjobs",
                      worker_job(now, **(worker_kw or {})), now=now, log=lambda *a, **k: None)
    stop_bucket, stop_worker = asyncio.Event(), asyncio.Event()
    sync = asyncio.create_task(bucket.run(stop_bucket))
    wtask = asyncio.create_task(run_worker(worker, stop_worker)) if start_worker else None
    if tamper:
        async def corrupt():
            for _ in range(2000):
                tars = list(bucket.worker_in.glob("trees/*.tar.gz"))
                if tars:
                    bucket.blocked.add("loop")  # keep the corrupted copy
                    tars[0].write_bytes(b"not a tarball")
                    return
                await asyncio.sleep(0.001)
        asyncio.create_task(corrupt())
    if kill_worker_at is not None:
        async def killer():
            while not (bucket.worker_out / "results").exists() or not any(
                    kill_worker_at in (json.loads(p.read_text()).get("marks") or {})
                    for p in bucket.worker_out.glob("results/*/progress.json")):
                await asyncio.sleep(0.002)
            stop_worker.set()
            wtask.cancel()
            bucket.blocked.add("worker")  # nothing more arrives from the dead VM
        asyncio.create_task(killer())
    seen = {}
    result, run = await asyncio.wait_for(host.split_pool_run("c0001", {"id": "j0001"}, decider(seen, cut_at),
                                                             ["s1", "s2", "s3"]), 20)
    await asyncio.sleep(0.05)
    stop_worker.set()
    stop_bucket.set()
    await sync
    if wtask:
        wtask.cancel()
    return result, seen, host, bucket, worker


def check_full(seen, marks=("s1", "s2", "s3")):
    for m in marks:
        assert sorted(seen[m]["games"]) == sorted(POOL), m
        for g in POOL:
            assert seen[m]["games"][g]["score"] == score(g, m)["score"]


# ----- unit: trees, mailbox, worker bookkeeping ---------------------------------------------------------------------
def test_delta_roundtrip_and_manifest(tmp_path, env):
    _, gen0, cand = env
    data, manifest = X.pack_delta(cand, gen0)
    out = X.unpack_delta(data, manifest, gen0, tmp_path / "out")
    assert X.tree_files(out) == X.tree_files(cand)
    assert not (out / "player/study.json").exists()
    bad = dict(manifest, **{"player/src/arena/new.py": "0" * 64})
    with pytest.raises(ValueError):
        X.unpack_delta(data, bad, gen0, tmp_path / "out2")


def test_mailbox_ignores_partial_and_foreign_json(tmp_path):
    box = X.Mailbox(tmp_path / "out", tmp_path / "peer")
    (tmp_path / "peer").mkdir()
    (tmp_path / "peer/hb.json").write_text('{"t": 1')
    assert box.read_json("hb.json") is None
    (tmp_path / "peer/hb.json").write_text("[1, 2]")
    assert box.read_json("hb.json") is None
    box.write_json("x.json", {"a": 1})
    assert box.read_json("x.json", mine=True) == {"a": 1}
    assert not list((tmp_path / "out").glob(".*"))  # no temporary files left behind


def test_half_plan_keeps_per_game_share():
    a, b = X.split_games(POOL)
    assert len(a) == 15 and len(b) == 14 and sorted(a + b) == sorted(POOL)
    wall, marks = X.half_plan(a, 273, FRACTIONS)
    assert wall == 15 * 273 and marks["s3"] == wall and marks["s2"] == round(wall * 150 / 380)


def test_worker_runs_each_job_once_in_posting_order(tmp_path, env):
    _, gen0, cand = env
    now = game_clock()
    disp = X.Dispatcher(X.Mailbox(tmp_path / "L", tmp_path / "Lin"), gen0, now=now)
    disp.post("j2", "c2", cand, ["g01"], 10, {"s3": 10})
    disp.post("j1", "c1", cand, ["g00"], 10, {"s3": 10})
    disp.cancel("j1", "cut")
    shutil.copytree(tmp_path / "L", tmp_path / "Win")
    ran = []

    async def run_job(job, tree, report, cancelled):
        ran.append(job["jid"])
        assert (tree / "player/src/arena/new.py").exists()
        return {"status": "complete", "marks": {"s3": {"games": {}}}}

    worker = X.Worker(X.Mailbox(tmp_path / "W", tmp_path / "Win"), gen0, tmp_path / "wj", run_job, now=now)
    assert asyncio.run(worker.step())
    assert ran == ["j2"]
    assert not asyncio.run(worker.step())  # j1 was cancelled before it started: final written, never run
    assert ran == ["j2"]
    assert json.loads((tmp_path / "W/results/j1/final.json").read_text())["status"] == "cancelled before start"
    # a restarted worker (same out folder) does not rerun finished jobs
    again = X.Worker(X.Mailbox(tmp_path / "W", tmp_path / "Win"), gen0, tmp_path / "wj", run_job, now=now)
    assert not asyncio.run(again.step())
    assert ran == ["j2"]


def test_worker_waits_for_incomplete_tree_and_reports_errors(tmp_path, env):
    _, gen0, cand = env
    now = game_clock()
    disp = X.Dispatcher(X.Mailbox(tmp_path / "L", tmp_path / "Lin"), gen0, now=now)
    disp.post("j1", "c1", cand, ["g00"], 10, {"s3": 10})
    (tmp_path / "Win/jobs").mkdir(parents=True)
    shutil.copy(tmp_path / "L/jobs/j1.json", tmp_path / "Win/jobs/j1.json")  # the job arrived, its tree not yet

    async def boom(job, tree, report, cancelled):
        raise RuntimeError("stack failed")

    worker = X.Worker(X.Mailbox(tmp_path / "W", tmp_path / "Win"), gen0, tmp_path / "wj", boom, now=now)
    assert not asyncio.run(worker.step())
    assert not (tmp_path / "W/results/j1/final.json").exists()
    shutil.copytree(tmp_path / "L/trees", tmp_path / "Win/trees")
    assert asyncio.run(worker.step())
    final = json.loads((tmp_path / "W/results/j1/final.json").read_text())
    assert final["status"].startswith("error: RuntimeError") and final["marks"] == {}


def test_dispatcher_liveness_and_lag(tmp_path, env):
    _, gen0, _ = env
    clock = [1000.0]
    box = X.Mailbox(tmp_path / "L", tmp_path / "Lin")
    disp = X.Dispatcher(box, gen0, now=lambda: clock[0])
    assert disp.worker_alive()  # not heard from yet, within BOOT of our start
    clock[0] += X.BOOT + 1
    assert not disp.worker_alive()
    (tmp_path / "Lin").mkdir()
    (tmp_path / "Lin/hb.json").write_text(json.dumps({"t": clock[0] - 10}))
    assert disp.worker_alive()
    # not started: posting time + STARTUP; started: the reported start
    assert not disp.lagging("j1", "s1", 100, posted=clock[0] - 100 - X.LATE - 1)
    assert disp.lagging("j1", "s1", 100, posted=clock[0] - 100 - X.LATE - X.STARTUP - 1)
    (tmp_path / "Lin/results/j2").mkdir(parents=True)
    (tmp_path / "Lin/results/j2/progress.json").write_text(json.dumps({"started": clock[0] - 100 - X.LATE - 1}))
    assert disp.lagging("j2", "s1", 100, posted=clock[0])


# ----- the loop and the worker over the mock bucket -----------------------------------------------------------------
def test_split_normal(env):
    tmp, gen0, cand = env
    result, seen, host, bucket, _ = asyncio.run(scenario(tmp, gen0, cand))
    check_full(seen)
    assert result["split"]["worker_mode"] == "worker" and result["cut"] is None and result["status"] == "complete"
    local = [r for r in FakeRun.made if r.tag.endswith("-a")]
    assert len(local) == 1 and sorted(local[0].games) == sorted(POOL[0::2])
    assert local[0].wall == 15 * 273
    job = json.loads((bucket.loop_out / "jobs/c0001-j0001-b.json").read_text())
    assert job["games"] == POOL[1::2] and job["wall"] == 14 * 273
    assert not any(r.tag.endswith("-b-here") for r in FakeRun.made)


def test_split_partial_json_in_transit(env, monkeypatch):
    tmp, gen0, cand = env
    monkeypatch.setattr(X, "HEARTBEAT", 0.012)  # a heartbeat outlives a sync, as on Nebius (30 s vs 60 s)
    result, seen, *_ = asyncio.run(scenario(tmp, gen0, cand, bucket_kw={"truncate_first": True}))
    check_full(seen)
    assert result["split"]["worker_mode"] == "worker"


def test_split_lost_uploads_are_resent(env):
    tmp, gen0, cand = env
    lose = ("c0001-j0001-b.json", "c0001-j0001-b.tar.gz", "progress.json", "final.json", "hb.json")
    result, seen, host, bucket, _ = asyncio.run(scenario(tmp, gen0, cand, bucket_kw={"lose": lose}))
    check_full(seen)
    assert result["split"]["worker_mode"] == "worker"
    assert len(bucket.lost) >= 4


def test_split_worker_dies_mid_job(env):
    tmp, gen0, cand = env
    result, seen, host, bucket, _ = asyncio.run(scenario(tmp, gen0, cand, kill_worker_at="s1"))
    check_full(seen)
    assert result["split"]["worker_mode"] == "local"
    assert any(e["event"] == "rsi2_split_takeback" for e in host.events)
    assert any(r.tag.endswith("-b-here") and sorted(r.games) == sorted(POOL[1::2]) for r in FakeRun.made)
    assert (bucket.loop_out / "cancel/c0001-j0001-b.json").exists()


def test_split_worker_never_starts(env):
    tmp, gen0, cand = env
    result, seen, host, *_ = asyncio.run(scenario(tmp, gen0, cand, start_worker=False))
    check_full(seen)
    assert result["split"]["worker_mode"] == "local"


def test_split_late_worker_is_taken_back_and_its_late_results_ignored(env):
    tmp, gen0, cand = env
    # the worker plays 6x slower than its marks: s1 lags past LATE, and its late scores (+50) must not be read
    result, seen, host, bucket, _ = asyncio.run(scenario(tmp, gen0, cand, worker_kw={"slow": 6.0, "delta": 50.0}))
    check_full(seen)
    assert result["split"]["worker_mode"] == "local"
    assert result["split"]["worker_failure"] == "silent or late"
    final = json.loads((bucket.worker_out / "results/c0001-j0001-b/final.json").read_text())
    assert final["cancelled"] and final["status"] == "stopped"


def test_split_extra_base_files_are_dropped(env):
    tmp, gen0, cand = env
    other = make_tree(tmp / "other0", {"player/src/arena/extra.py": "z\n"})  # not in the manifest: removed
    result, seen, *_ = asyncio.run(scenario(tmp, gen0, cand, worker_base=other))
    check_full(seen)
    assert result["split"]["worker_mode"] == "worker"


def test_split_tree_too_big_plays_both_halves_here(env, monkeypatch):
    tmp, gen0, cand = env
    monkeypatch.setattr(X, "MAX_TREE", 10)
    result, seen, host, bucket, _ = asyncio.run(scenario(tmp, gen0, cand))
    check_full(seen)
    assert result["split"]["worker_failure"] == "tree too big to send"
    assert not (bucket.loop_out / "jobs").exists()


def test_split_cut_cancels_the_worker(env):
    tmp, gen0, cand = env
    result, seen, host, bucket, _ = asyncio.run(scenario(tmp, gen0, cand, cut_at="s2"))
    assert result["cut"] == "s2" and "s3" not in seen
    check_full(seen, ("s1", "s2"))
    local = next(r for r in FakeRun.made if r.tag.endswith("-a"))
    assert local.stopped == "cut at s2"
    final = json.loads((bucket.worker_out / "results/c0001-j0001-b/final.json").read_text())
    assert final["cancelled"] and "s3" not in final["marks"]


def test_split_worker_with_a_different_gen0_reports_bad_tree(env, tmp_path):
    tmp, gen0, cand = env
    other = make_tree(tmp / "other0", {"serving/server.json": '{"x": 1}'})  # a file the delta does not carry
    result, seen, host, bucket, _ = asyncio.run(scenario(tmp, gen0, cand, worker_base=other))
    check_full(seen)
    assert result["split"]["worker_mode"] == "local"
    assert result["split"]["worker_failure"].startswith("bad tree")


def test_split_tampered_tree_never_runs_and_is_taken_back(env):
    tmp, gen0, cand = env
    result, seen, host, bucket, _ = asyncio.run(scenario(tmp, gen0, cand, tamper=True))
    check_full(seen)
    assert result["split"]["worker_mode"] == "local"
    assert not (bucket.worker_out / "results/c0001-j0001-b/final.json").exists()


# ----- split_mode "parallel": the worker VM evaluates other candidates on the whole pool ----------------------------
WORKER_OFFSET = 5.0  # the worker VM scores every game 5 points higher (a VM effect its own reference absorbs)


class ParallelRun(FakeRun):
    deltas = {}

    def __init__(self, host, cid, tree, **kw):
        super().__init__(host, cid, tree, **kw)
        self.delta = ParallelRun.deltas.get(cid, 0.0)


def make_loop(tmp, bucket, gen0, cand, now, cands, deltas, monkeypatch, refs=2):
    monkeypatch.setattr(L, "Run", ParallelRun)
    ParallelRun.deltas = dict(deltas)
    lp = L.Loop.__new__(L.Loop)
    lp.cfg = {**L.DEFAULTS, "pool": POOL, "split_peer": "w", "split_poll": 0.003, "root": str(tmp / "root")}
    lp.pool = POOL
    lp.cfg["marks"] = {m: round(29 * 273 * f) for m, f in FRACTIONS.items()}
    lp.root = tmp / "root"
    lp.deadline = time.time() + 10 ** 6
    lp.dispatcher = X.Dispatcher(X.Mailbox(bucket.loop_out, bucket.loop_in), gen0, now=now)
    lp.runtime = type("RT", (), {"seal": lambda self, p: None})()
    lp.proposer = None
    lp.events_rows = []
    lp.event_log = lambda row: lp.events_rows.append(row)
    lp.save = lambda reason="": None
    lp.stop_flag = False
    lp.stop_requested = lambda: lp.stop_flag
    lp.job_task = lp.current_job = lp.remote_task = lp.remote_job = None
    lp.stopping = False
    lp.started = time.time()
    ref = {g: {**{m: score(g, m)["score"] for m in FRACTIONS}, **{f"{m}_c": 10.0 for m in FRACTIONS}} for g in POOL}
    lp.state = {"seq": 100, "meters": {"loop_s": 0.0, "legs": 1}, "candidates": {}, "jobs": [], "runs": [],
                "practice": {}, "stack": {"live": "c0000", "live_hash": "h0", "history": []},
                "meta": {"live": None, "items": {}}, "proposer": {"sessions": []}, "accepted": [],
                "cont_ref_queued": True}
    for _ in range(refs):
        lp.add_reference(json.loads(json.dumps(ref)))
    trees = {"c0000": gen0, **{c: cand for c in cands}}
    lp.candidate_dir = lambda cid: trees[cid]
    for cid in ["c0000", *cands]:
        lp.state["candidates"][cid] = {"parent": None, "status": "queued", "marks": {}, "problems": [],
                                       "serving_hash": "h0"}

    async def ok(cid):
        return True
    lp.ensure_stack = ok
    return lp


def remote_job(now, deltas, slow=1.0):
    async def run_job(job, tree, report, cancelled):
        return await worker_job(now, slow=slow, delta=WORKER_OFFSET + deltas.get(job["cid"], 0.0))(
            job, tree, report, cancelled)
    return run_job


async def drive(lp, until, limit=30):
    t0 = time.time()
    while not until():
        lp.launch()
        lp.launch_remote()
        await asyncio.sleep(0.003)
        assert time.time() - t0 < limit, "loop did not finish"


async def parallel(tmp, gen0, cand, monkeypatch, cands, deltas, *, queue=None, kill_after=None, stop_after=None,
                   slow=1.0):
    now = game_clock()
    bucket = Bucket(tmp)
    lp = make_loop(tmp, bucket, gen0, cand, now, cands, deltas, monkeypatch)
    lp.state["jobs"] = queue or [{"id": f"j{i}", "kind": "eval", "candidate": c, "queued": 0} for i, c in
                                 enumerate(cands)]
    worker = X.Worker(X.Mailbox(bucket.worker_out, bucket.worker_in), gen0, tmp / "wjobs",
                      remote_job(now, deltas, slow=slow), now=now, log=lambda *a, **k: None)
    stop_bucket, stop_worker = asyncio.Event(), asyncio.Event()
    sync = asyncio.create_task(bucket.run(stop_bucket))
    wtask = asyncio.create_task(run_worker(worker, stop_worker))
    if kill_after:
        async def killer():
            while not any(e.get("event") == "rsi2_mark" and e.get("vm") == "worker" and e["candidate"] != "c0000"
                          for e in lp.events_rows):
                await asyncio.sleep(0.002)
            stop_worker.set()
            wtask.cancel()
            bucket.blocked.add("worker")
        asyncio.create_task(killer())
    if stop_after:
        async def stopper():
            while not any(e.get("event") == "rsi2_remote_post" and e["candidate"] != "c0000"
                          for e in lp.events_rows):
                await asyncio.sleep(0.002)
            lp.stop_flag = True
        asyncio.create_task(stopper())

    def done():
        busy = any(t is not None and not t.done() for t in (lp.job_task, lp.remote_task))
        if stop_after and lp.stop_flag:
            return not busy
        return not busy and not lp.state["jobs"] and all(
            lp.state["candidates"][c]["status"] not in ("queued",) for c in cands) and (
            lp.state.get("vm_refs", {}).get("worker", {}).get("reference") or lp.state.get("remote_disabled")
            or kill_after)
    await drive(lp, done)
    stop_worker.set()
    stop_bucket.set()
    await sync
    wtask.cancel()
    return lp, bucket


def verdicts(lp, cid):
    return {m: r.get("diff") for m, r in lp.state["candidates"][cid]["marks"].items()}


def test_parallel_two_vms_each_against_its_own_reference(env, monkeypatch):
    tmp, gen0, cand = env
    cands = ["c0001", "c0002", "c0003"]
    lp, bucket = asyncio.run(parallel(tmp, gen0, cand, monkeypatch, cands, {"c0001": 2.0, "c0002": 2.0, "c0003": 2.0}))
    wref = lp.state["vm_refs"]["worker"]["reference"]
    assert wref["g00"]["s3"] == score("g00", "s3")["score"] + WORKER_OFFSET  # the worker's gen-0 run
    vms = {c: lp.state["candidates"][c].get("vm") for c in cands}
    assert set(vms.values()) == {"loop", "worker"}, vms
    for c in cands:  # the VM effect cancels against each VM's own reference
        item = lp.state["candidates"][c]
        assert item["status"] == "done" and item["verdict"] == "reject"  # cont equal: no accept
        assert all(abs(d - 2.0) < 1e-9 for d in verdicts(lp, c).values()), (c, verdicts(lp, c))
    starts = [e for e in lp.events_rows if e["event"] == "rsi2_job_start" and e.get("vm") == "worker"]
    assert starts[0]["candidate"] == "c0000"  # the worker's own gen-0 run comes first there


def test_parallel_cut_on_the_worker_cancels_it(env, monkeypatch):
    tmp, gen0, cand = env
    lp, bucket = asyncio.run(parallel(tmp, gen0, cand, monkeypatch, ["c0001", "c0002"],
                                      {"c0001": 0.0, "c0002": -30.0}))
    item = lp.state["candidates"]["c0002"]
    assert item["verdict"] == "cut at s2", item
    if item["vm"] == "worker":
        final = json.loads(next(bucket.worker_out.glob("results/c0002-*/final.json")).read_text())
        assert final["cancelled"] and "s3" not in final["marks"]


def test_parallel_worker_dies_job_requeued_here(env, monkeypatch):
    tmp, gen0, cand = env
    monkeypatch.setattr(X, "DEAD", 300)
    cands = ["c0001", "c0002", "c0003"]
    lp, bucket = asyncio.run(parallel(tmp, gen0, cand, monkeypatch, cands, {c: 1.0 for c in cands},
                                      kill_after=True))
    lost = [e for e in lp.events_rows if e["event"] == "rsi2_worker_lost"]
    assert len(lost) == 1 and lost[0]["candidate"] != "c0000"
    assert lost[0]["why"] == "worker silent" or lost[0]["why"].startswith("worker late")
    assert lp.state["candidates"][lost[0]["candidate"]]["vm"] == "loop"  # rerun here from the start
    for c in cands:
        item = lp.state["candidates"][c]
        assert item["status"] == "done", (c, item["status"])
        assert all(abs(d - 1.0) < 1e-9 for d in verdicts(lp, c).values())
    assert lp.state["remote_failures"] == 1


def test_parallel_worker_takes_an_eval_behind_practice(env, monkeypatch):
    tmp, gen0, cand = env
    queue = [{"id": "j1", "kind": "practice", "candidate": "c0001", "games": ["g00"], "seconds": 2000, "queued": 0},
             {"id": "j2", "kind": "eval", "candidate": "c0001", "queued": 0}]
    now = game_clock()
    bucket = Bucket(tmp)
    lp = make_loop(tmp, bucket, gen0, cand, now, ["c0001"], {}, monkeypatch)
    lp.state["vm_refs"] = {"worker": {"references": [{}], "reference": {}}}  # worker reference already there
    lp.state["jobs"] = queue
    lp.job_task = type("Busy", (), {"done": lambda self: False})()  # the loop VM is busy
    taken = []

    async def fake(job, vm="loop"):
        taken.append((job["id"], vm))
    lp.run_job = fake

    async def go():
        lp.launch_remote()
        await asyncio.sleep(0)
    asyncio.run(go())
    assert taken == [("j2", "worker")] and [j["id"] for j in lp.state["jobs"]] == ["j1"]


def test_parallel_loop_stop_requeues_the_remote_job(env, monkeypatch):
    tmp, gen0, cand = env
    lp, bucket = asyncio.run(parallel(tmp, gen0, cand, monkeypatch, ["c0001", "c0002", "c0003", "c0004"], {},
                                      stop_after=True))
    lost = [e for e in lp.events_rows if e["event"] == "rsi2_worker_lost"]
    assert lost and lost[-1]["why"] == "loop stop"
    assert any(j["candidate"] == lost[-1]["candidate"] and j.get("local_only") for j in lp.state["jobs"])
    assert (bucket.loop_out / f"cancel/{lost[-1]['candidate']}-{lost[-1]['job']}-w.json").exists()
    assert lp.state.get("remote_failures", 0) == 0  # a stop is not the worker's fault


def test_resume_restores_big_files_and_flags_changed_ones(tmp_path, env, monkeypatch):
    _, gen0, cand = env
    from rsi2_loop import candidates as C
    src = tmp_path / "bundle"
    shutil.copytree(gen0, src / "rsi/gen0/tree")
    (src / "rsi/gen0/tree/serving/big.bin").write_bytes(b"x" * 100)
    lp = L.Loop.__new__(L.Loop)
    lp.source, lp.state, rows = src, {"candidates": {}, "jobs": []}, []
    lp.event_log = rows.append
    trees = {}
    lp.candidate_dir = lambda cid: trees[cid]
    for cid, big in (("c1", b"x" * 100), ("c2", b"y" * 100)):
        full = tmp_path / cid
        shutil.copytree(src / "rsi/gen0/tree", full)
        (full / "serving/big.bin").write_bytes(big)
        lp.state["candidates"][cid] = {"status": "queued", "problems": [], "serving_hash": C.serving_hash(full)}
        (full / "serving/big.bin").unlink()  # what the bucket checkpoint holds
        trees[cid] = full
        lp.state["jobs"].append({"id": "j" + cid, "kind": "eval", "candidate": cid})
    for cid in ("c1", "c2"):
        lp.restore_big_files(cid, limit=50)
    assert (trees["c1"] / "serving/big.bin").read_bytes() == b"x" * 100
    assert lp.state["candidates"]["c1"]["status"] == "queued"
    assert lp.state["candidates"]["c2"]["status"] == "error"  # its own big file was lost: not run as gen 0's
    assert [j["candidate"] for j in lp.state["jobs"]] == ["c1"]


def test_worker_host_run_job_reports_start_marks_and_cancel(tmp_path, env, monkeypatch):
    _, gen0, cand = env
    from rsi2_loop import worker as W
    monkeypatch.setattr(W, "Run", FakeRun)
    real_sleep = asyncio.sleep
    monkeypatch.setattr(W.asyncio, "sleep", lambda s: real_sleep(min(s, 0.002)))
    host = W.WorkerHost.__new__(W.WorkerHost)
    host.root, host.events_rows = tmp_path, []
    host.event_log = host.events_rows.append

    async def same_stack(cid, tree):
        return None
    host.ensure_stack = same_stack
    reports = []
    job = {"jid": "c1-j1-w", "cid": "c1", "games": POOL[:3], "wall": 4000, "marks": {"s1": 500, "s2": 1500, "s3": 4000}}
    final = asyncio.run(host.run_job(job, cand, reports.append, lambda: False))
    assert final["status"] == "complete" and sorted(final["marks"]) == ["s1", "s2", "s3"]
    assert any(r.get("started") for r in reports)
    flag = {"cancel": False}

    async def cut_later():
        task = asyncio.create_task(host.run_job(job, cand, reports.append, lambda: flag["cancel"]))
        await real_sleep(0.08)
        flag["cancel"] = True
        return await task
    final = asyncio.run(cut_later())
    assert final["cancelled"] and final["status"] == "stopped" and "s3" not in final["marks"]


# ----- practice batches ------------------------------------------------------------------
def practice_loop(tmp_path, jobs, serving=None):
    lp = L.Loop.__new__(L.Loop)
    lp.cfg = {**L.DEFAULTS, "pool": POOL}
    lp.cfg["marks"] = {m: round(29 * 273 * f) for m, f in FRACTIONS.items()}
    lp.state = {"jobs": jobs, "candidates": {}, "runs": [], "practice": {}, "reference": {"x": 1}}
    for j in jobs:
        lp.state["candidates"].setdefault(j["candidate"], {"serving_hash": (serving or {}).get(j["candidate"], "h0"),
                                                         "status": "queued", "problems": [], "marks": {}})
    lp.deadline = time.time() + 10 ** 6
    lp.job_task = lp.current_job = lp.remote_task = lp.remote_job = None
    lp.stopping, lp.dispatcher = False, None
    lp.rows = []
    lp.event_log = lp.rows.append
    lp.save = lambda reason="": None
    return lp


def pj(i, cid, games=("a", "b"), seconds=546):
    return {"id": f"j{i}", "kind": "practice", "candidate": cid, "games": list(games), "seconds": seconds}


def test_practice_batch_keeps_ab_arms_together(tmp_path):
    jobs = [pj(1, "c0000"), pj(2, "c1", ("x",), 273), pj(3, "c2"), pj(4, "c3", ("y",), 273), pj(5, "c4", ("y",), 273)]
    lp = practice_loop(tmp_path, jobs)
    assert lp.practice_batch(0) == [0, 1, 2]  # the (a, b) A/B pair, then (x) fits whole; the (y) pair would not
    jobs2 = [pj(1, "c0000"), pj(2, "c1", ("x",), 273), pj(3, "c2")]
    assert practice_loop(tmp_path, jobs2).practice_batch(0) == [0, 1, 2]
    # arms whose serving differs: each alone
    lp3 = practice_loop(tmp_path, [pj(1, "c0000"), pj(2, "c9")], serving={"c9": "h9"})
    assert lp3.practice_batch(0) == [0]
    lp1 = practice_loop(tmp_path, [pj(1, "c0000"), pj(2, "c2")])
    lp1.cfg["practice_parallel"] = 1
    assert lp1.practice_batch(0) == [0]


def test_pool_evaluations_start_before_practice(tmp_path):
    jobs = [pj(1, "c0000"), pj(2, "c2"), {"id": "j3", "kind": "eval", "candidate": "c5"}]
    lp = practice_loop(tmp_path, jobs)
    lp.state["cont_ref_queued"] = True
    lp.cfg["continuous"] = False
    started = []

    async def fake_job(job, vm="loop"):
        started.append(job["id"])

    async def fake_batch(batch):
        started.append([j["id"] for j in batch])
    lp.run_job, lp.run_batch = fake_job, fake_batch

    async def go():
        lp.launch()
        await asyncio.sleep(0)
        lp.job_task = None
        lp.launch()
        await asyncio.sleep(0)
    asyncio.run(go())
    assert started == ["j3", ["j1", "j2"]]
    assert lp.state["jobs"] == []


class PracticeRun:
    live = 0
    peak = 0
    shares = []

    def __init__(self, host, cid, tree, *, kind, games, games_dir, wall, tag, **_):
        self.cid, self.games, self.wall = cid, games, wall
        self.out = Path("/nonexistent")
        self.mem_share = 1.0

    async def go(self, decide=None):
        PracticeRun.live += 1
        PracticeRun.peak = max(PracticeRun.peak, PracticeRun.live)
        PracticeRun.shares.append(self.mem_share)
        await asyncio.sleep(0.05)
        PracticeRun.live -= 1
        return {"status": "complete", "started": time.time(), "marks": {}}

    def scores_at(self, seconds, started):
        return {"games": {g: {"levels": 1} for g in self.games}}


def test_run_batch_plays_at_once_and_records_partners(tmp_path, monkeypatch):
    monkeypatch.setattr(L, "Run", PracticeRun)
    PracticeRun.live = PracticeRun.peak = 0
    PracticeRun.shares = []
    jobs = [pj(1, "c0000"), pj(2, "c2")]
    lp = practice_loop(tmp_path, [])
    lp.state["candidates"] = {c: {"serving_hash": "h0", "status": "queued", "problems": []} for c in ("c0000", "c2")}
    lp.root = tmp_path
    lp.candidate_dir = lambda cid: tmp_path / cid
    lp.runtime = type("RT", (), {"seal": lambda self, p: None})()
    lp.proposer = None

    async def ok(cid):
        return True
    lp.ensure_stack = ok
    asyncio.run(lp.run_batch(jobs))
    assert PracticeRun.peak == 2 and PracticeRun.shares == [0.5, 0.5]
    res = json.loads((tmp_path / "archive/c2/practice/j2/result.json").read_text())
    assert res["played_at_once_with"] == ["j1"]
    assert len(lp.state["runs"]) == 2


def test_worker_plays_a_practice_batch_and_returns_traces(env, monkeypatch, tmp_path):
    tmp, gen0, cand = env
    from rsi2_loop import worker as W
    now = game_clock()
    bucket = Bucket(tmp)
    lp = make_loop(tmp, bucket, gen0, cand, now, ["c0001"], {}, monkeypatch)
    lp.state["vm_refs"] = {"worker": {"references": [{}], "reference": {}}}
    lp.job_task = type("Busy", (), {"done": lambda self: False})()  # the loop VM is busy
    lp.state["jobs"] = [pj(1, "c0000", ("g1", "g2"), 500), pj(2, "c0001", ("g1", "g2"), 500)]
    lp.state["candidates"]["c0000"]["serving_hash"] = lp.state["candidates"]["c0001"]["serving_hash"] = "h0"
    starts = []
    wbox = X.Mailbox(bucket.worker_out, bucket.worker_in)

    async def run_job(job, tree, report, cancelled):
        starts.append((job["jid"], time.time(), job["share"]))
        report({"started": now(), "marks": {}})
        await asyncio.sleep(500 * SCALE)
        folder = tmp / "traces" / job["jid"]
        (folder / "k000").mkdir(parents=True, exist_ok=True)
        (folder / "k000/log.jsonl").write_text(job["cid"])
        (folder / "k000/state.sqlite").write_text("db")
        data = W.pack_traces(folder)
        wbox.write(f"results/{job['jid']}/practice.tar.gz", data)
        return {"status": "complete", "marks": {}, "practice": {"scores": {"g1": {"levels": 2}},
                                                                "sha": X.sha_bytes(data), "bytes": len(data)}}
    worker = X.Worker(wbox, gen0, tmp / "wjobs", run_job, now=now, log=lambda *a, **k: None)

    async def go():
        stop = asyncio.Event()
        sync = asyncio.create_task(bucket.run(stop))
        wtask = asyncio.create_task(run_worker(worker, stop))
        lp.launch_remote()
        await asyncio.wait_for(lp.remote_task, 20)
        stop.set()
        await sync
        wtask.cancel()
    asyncio.run(go())
    assert len(starts) == 2 and abs(starts[0][1] - starts[1][1]) < 0.05 and starts[0][2] == 0.5  # together
    assert L.games_share(pj(1, "a", ("g",) * 12), [pj(1, "a", ("g",) * 12), pj(2, "b", ("g",) * 4)]) == 0.75
    for cid, jid in (("c0000", "j1"), ("c0001", "j2")):
        target = lp.root / "archive" / cid / "practice" / jid
        res = json.loads((target / "result.json").read_text())
        assert res["machine"] == "worker" and res["played_at_once_with"] == [{"j1": "j2", "j2": "j1"}[jid]]
        assert (target / "games/k000/log.jsonl").read_text() == cid
        assert not (target / "games/k000/state.sqlite").exists()
    assert lp.state["jobs"] == [] and lp.state["practice"]["g1"]


def test_worker_practice_lost_goes_back_to_the_queue(env, monkeypatch):
    tmp, gen0, cand = env
    now = game_clock()
    bucket = Bucket(tmp)
    bucket.blocked.add("worker")  # nothing ever comes back
    lp = make_loop(tmp, bucket, gen0, cand, now, ["c0001"], {}, monkeypatch)
    lp.state["vm_refs"] = {"worker": {"references": [{}], "reference": {}}}
    lp.job_task = type("Busy", (), {"done": lambda self: False})()
    lp.state["jobs"] = [pj(1, "c0000", ("g1",), 300), pj(2, "c0001", ("g1",), 300)]
    monkeypatch.setattr(X, "BOOT", 50)

    async def go():
        stop = asyncio.Event()
        sync = asyncio.create_task(bucket.run(stop))
        lp.launch_remote()
        await asyncio.wait_for(lp.remote_task, 20)
        stop.set()
        await sync
    asyncio.run(go())
    assert [j["id"] for j in lp.state["jobs"]] == ["j1", "j2"] and all(j["local_only"] for j in lp.state["jobs"])
    assert (bucket.loop_out / "cancel/c0001-j2-w.json").exists()


def test_worker_host_practice_writes_traces(tmp_path, env, monkeypatch):
    _, gen0, cand = env
    from rsi2_loop import worker as W
    monkeypatch.setattr(W, "Run", PracticeRun)
    host = W.WorkerHost.__new__(W.WorkerHost)
    host.root, rows = tmp_path, []
    host.event_log = rows.append
    host.box = X.Mailbox(tmp_path / "out", tmp_path / "peer")

    async def same_stack(cid, tree):
        return None
    host.ensure_stack = same_stack
    job = {"jid": "c1-j1-w", "cid": "c1", "kind": "practice", "games": ["g1"], "wall": 300, "marks": {}, "share": 0.5}
    final = asyncio.run(host.run_job(job, cand, lambda d: None, lambda: False))
    data = (tmp_path / "out/results/c1-j1-w/practice.tar.gz").read_bytes()
    assert final["practice"]["sha"] == X.sha_bytes(data) and final["practice"]["scores"] == {"g1": {"levels": 1}}
    assert PracticeRun.shares[-1] == 0.5
    big = tmp_path / "big"
    big.mkdir()
    (big / "a.txt").write_text("a")
    (big / "b.bin").write_bytes(os.urandom(5000))
    small = W.pack_traces(big, limit=3000)  # the largest file is left out above the limit
    import io as _io, tarfile as _tf
    assert _tf.open(fileobj=_io.BytesIO(small)).getnames() == ["a.txt"]


# ----- serving switch path (l8w, 18 Oct: absolute mask path refused, view missing helpers) ---------------------------
def test_stage_study_keeps_mask_relative_under_the_study_grandparent(tmp_path):
    from rsi2_loop import serving as S
    serving = tmp_path / "cand/serving"
    (serving / "expert-prune").mkdir(parents=True)
    (serving / "expert-prune/m.json").write_text("{}")
    (serving / "server.json").write_text(json.dumps({"expert_prune": {"mask": "expert-prune/m.json", "sha256": "0" * 64},
                                                     "draft_head": {"overlay": {"path": "draft-head/o"}}}))
    st = S.Stacks.__new__(S.Stacks)
    st.base_study = {"server": {}, "generation_slots": 4}
    path = st.stage_study(serving, tmp_path / "stage", serving.resolve())
    study = json.loads(path.read_text())
    assert path == tmp_path / "stage/configs/study.json"
    assert study["server"]["expert_prune"]["mask"] == "expert-prune/m.json"
    assert (path.parent.parent / study["server"]["expert_prune"]["mask"]).read_text() == "{}"
    assert study["server"]["draft_head"]["overlay"]["path"] == "draft-head/o"


def test_loop_stack_down_stops_the_leg_and_keeps_the_candidate(tmp_path):
    lp = practice_loop(tmp_path, [])
    lp.state["candidates"]["c0087"] = {"serving_hash": "h1", "status": "evaluating", "problems": [], "marks": {}}
    lp.state["candidates"]["c0000"] = {"serving_hash": "h0", "status": "accepted", "problems": [], "marks": {}}
    lp.state["stack"] = {"live": "c0000", "live_hash": "h0", "history": []}
    lp.output = tmp_path
    lp.candidate_dir = lambda cid: tmp_path / cid
    lp.notify = lambda row: lp.rows.append(row)
    lp.proxy = type("P", (), {"open": asyncio.Event(), "inflight": {}})()

    async def dead(*a, **k):
        raise RuntimeError("start c0000-back: exit 1")
    lp.stacks = type("St", (), {"switch": staticmethod(dead)})()
    job = {"id": "j0088", "kind": "eval", "candidate": "c0087", "local_only": True}
    asyncio.run(lp.run_job(job))
    assert lp.state["jobs"] == [{"id": "j0088", "kind": "eval", "candidate": "c0087"}]
    assert lp.state["candidates"]["c0087"]["status"] == "queued" and not lp.state["candidates"]["c0087"]["problems"]
    assert (tmp_path / "stop-request").exists()
    assert any(r.get("event") == "rsi2_stack_down" for r in lp.rows)
    assert not any(r.get("event") in ("candidate_rejected", "rsi2_job_error") for r in lp.rows)


def worker_host_with_switch(tmp_path, gen0, switch):
    from rsi2_loop import worker as W
    from rsi2_loop import candidates as C
    host = W.WorkerHost.__new__(W.WorkerHost)
    host.root, host.events_rows, host.gen0, host.dead = tmp_path / "w", [], gen0, None
    host.event_log = host.events_rows.append
    host.live = ("c0000", C.serving_hash(gen0), gen0 / "serving")
    host.stacks = type("St", (), {"switch": staticmethod(switch)})()
    return host


def test_worker_selftest_runs_both_paths(tmp_path, env):
    _, gen0, _ = env
    (gen0 / "serving").mkdir(exist_ok=True)
    (gen0 / "serving/server.json").write_text("{}")
    calls = []

    async def switch(new_dir, new_tag, old_dir, old_tag, fallback=None):
        calls.append((new_tag, old_tag))
        return {"ok": True, "live": "new"}
    host = worker_host_with_switch(tmp_path, gen0, switch)
    host.cfg = {"selftest_untrusted": True}
    asyncio.run(host.selftest())
    assert calls == [("selftest", "c0000"), ("c0000-selftest", "c0000")] and host.live[0] == "c0000" and not host.dead
    calls.clear()
    host = worker_host_with_switch(tmp_path, gen0, switch)
    host.cfg = {}
    asyncio.run(host.selftest())
    assert calls == [("c0000-selftest", "c0000")] and not host.dead
    assert host.events_rows[-1]["event"] == "rsi2w_selftest" and host.events_rows[-1]["ok"]

    async def broken(new_dir, new_tag, old_dir, old_tag, fallback=None):
        raise RuntimeError("nothing came back")
    host = worker_host_with_switch(tmp_path, gen0, broken)
    host.cfg = {"selftest_untrusted": True}
    try:
        asyncio.run(host.selftest())
    except RuntimeError:
        pass
    assert host.dead and any(r["event"] == "rsi2w_stack_down" for r in host.events_rows)


# ----- meter, practice withdraw/floor, worker without a loop ------------------------------
def test_withdraw_practice_by_id_and_per_game_floor(tmp_path):
    lp = practice_loop(tmp_path, [pj(1, "c1"), pj(2, "c0000"), {"id": "j3", "kind": "eval", "candidate": "c1"}])
    assert lp.withdraw(job="j2") == {"withdrawn": True, "job": "j2"}
    assert lp.withdraw(job="j9")["withdrawn"] is False
    assert lp.withdraw("c1") == {"withdrawn": True, "jobs": ["j1", "j3"]} and lp.state["jobs"] == []
    assert lp.state["candidates"]["c1"]["status"] == "withdrawn"
    lp.practice_games = lambda: [f"g{i}" for i in range(12)]
    lp.new_id = lambda p: "j10"
    try:
        lp.request_practice("c0000", [f"g{i}" for i in range(12)], 12, 272.7)
        raise AssertionError("12 games in 272.7 s must be refused")
    except ValueError as exc:
        assert "per game" in str(exc)
    assert lp.request_practice("c0000", ["g0", "g1", "g2", "g3"], 4, 272.7)["queued"] == "j10"


def test_worker_meter_counts_only_while_alive(tmp_path):
    lp = practice_loop(tmp_path, [])
    lp.state["meters"] = {"loop_s": 0.0}
    alive = {"v": True}
    lp.dispatcher = type("D", (), {"worker_alive": lambda self: alive["v"]})()
    lp._worker_tick = time.time() - 100
    lp.meter_worker()
    assert 99 <= lp.state["meters"]["worker_s"] <= 102
    alive["v"] = False
    lp._worker_tick = time.time() - 100
    lp.meter_worker()
    assert lp.state["meters"]["worker_s"] < 103


def test_worker_loop_alive(tmp_path):
    t = {"now": 1000.0}
    box = X.Mailbox(tmp_path / "w_out", tmp_path / "l_out")
    (tmp_path / "l_out").mkdir(parents=True)
    w = X.Worker(box, tmp_path, tmp_path / "wj", None, now=lambda: t["now"], log=lambda *a: None)
    assert w.loop_alive(since=900.0)
    t["now"] = 900.0 + X.LOOP_WAIT + 1
    assert not w.loop_alive(since=900.0)
    (tmp_path / "l_out/loop.json").write_text(json.dumps({"t": t["now"] - 10}))
    assert w.loop_alive(since=900.0)


def test_hold_serving_changes_skips_their_jobs(tmp_path):
    lp = practice_loop(tmp_path, [{"id": "j1", "kind": "eval", "candidate": "c9"}, pj(2, "c9"), pj(3, "c0000"),
                                  {"id": "j4", "kind": "eval", "candidate": "c1"}], serving={"c9": "h9"})
    lp.state["candidates"]["c0000"] = {"serving_hash": "h0", "status": "accepted", "problems": [], "marks": {}}
    assert lp.local_index() == 0
    lp.cfg["hold_serving_changes"] = True
    assert lp.local_index() == 3
    lp.state["jobs"] = lp.state["jobs"][:2]
    assert lp.local_index() is None
