"""One arena run of a candidate (fixed side): the pool evaluation with racing, or a practice play.

The candidate's ``player/`` (harness source, configs, study keys) is assembled with gen 0's fixed parts into a run
folder: the fixed study keys (roster, wall, marks, competition block, trials) and the fixed files (``FIXED_FILES``:
the arena runner, the competition gateway client, early stop, live progress, the arcade wrapper) always come from gen
0, so a candidate cannot change the yardstick. Games are served by the fixed gateway process (``rsi2_loop.gateway``,
root, outside the sandbox); the candidate's ``arena.run`` plays them over loopback as user ``rsieval`` in its own
mount namespace (no game files, no loop folders, no checkpoint), and the scores at the marks come from the gateway's
snapshots, never from files the candidate wrote.

Racing (docs/rsi.md 3, restart; bars from the 17 Oct noise check): the pool run continues through the marks while
its paired difference to gen 0's per-game reference stays above the cut at s1 and s2; at s3 it is accepted when the
difference reaches ``accept``. A cut writes ``stop-request`` into the trial folder (``arena.early_stop``), so the run
ends gracefully at that mark.
"""
from __future__ import annotations

import asyncio
import json
import os
import shutil
import signal
import socket
import subprocess
import sys
import time
from pathlib import Path

from . import sandbox

FIXED_FILES = ("arena/run.py", "arena/competition.py", "arena/early_stop.py", "arena/live_progress.py",
               "arena/report.py", "arc_infra/arcade.py", "taaf/competition_arcade.py")
FORBIDDEN_PACKAGES = ("arc_agi", "arcengine", "rsi2_loop", "rsi_loop", "sitecustomize.py", "usercustomize.py")
FIXED_KEYS = ("games", "generated_games", "competition", "wall_seconds", "notebook_seconds", "live_progress", "rsi2",
              "name", "trials", "server", "model_datasets", "early_stop", "smoke", "status", "kernel_timeout",
              "served_model")
EVAL_USER = "rsieval"


def proposer_load(loop):
    """Requests and tokens the proposer and check sessions sent through the metering proxy so far (None without one)."""
    totals = getattr(getattr(loop, "proxy", None), "totals", None)
    if totals is None:
        return None
    keys = ("requests", "prompt_tokens", "completion_tokens", "cached_tokens")
    return {k: sum(int(t.get(k, 0)) for s, t in list(totals.items()) if s != "_unknown") for k in keys}


def free_port():
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def tree_problems(tree: Path) -> list[str]:
    """Layout problems of a candidate tree (before any GPU time)."""
    problems = []
    for rel in ("player/src/arena", "player/configs/duck_controls.yaml", "player/study.json", "serving/server.json"):
        if not (tree / rel).exists():
            problems.append(f"missing {rel}")
    src = tree / "player/src"
    if src.is_dir():
        for name in FORBIDDEN_PACKAGES:
            if (src / name).exists():
                problems.append(f"player/src/{name}: not allowed (engine, scorer or loop code)")
    for rel in ("player/study.json", "serving/server.json"):
        try:
            data = json.loads((tree / rel).read_text())
            if not isinstance(data, dict):
                problems.append(f"{rel} must be an object")
            elif rel == "player/study.json":
                fixed = sorted(k for k in data if k in FIXED_KEYS)
                if fixed:
                    problems.append(f"player/study.json sets fixed keys {fixed} (they come from gen 0)")
        except (OSError, ValueError) as exc:
            problems.append(f"{rel}: {exc}"[:300])
    return problems


def compose(bundle: Path, tree: Path, run: Path, *, games, games_wall, extra=None):
    """The run folder: candidate source + gen 0's fixed files, configs, and the composed study."""
    base = json.loads((bundle / "configs/comparison.json").read_text())
    if run.exists():
        shutil.rmtree(run)
    run.mkdir(parents=True)
    shutil.copytree(tree / "player/src", run / "src", symlinks=False,
                    ignore=shutil.ignore_patterns("__pycache__", *FORBIDDEN_PACKAGES))
    for rel in FIXED_FILES:
        target = run / "src" / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(bundle / "src" / rel, target)
    shutil.copytree(tree / "player/configs", run / "configs", symlinks=False)
    study = json.loads(json.dumps(base))
    study.update({k: v for k, v in json.loads((tree / "player/study.json").read_text()).items()
                  if k not in FIXED_KEYS})
    study["server"] = json.loads((tree / "serving/server.json").read_text())
    study["games"] = list(games)
    study.pop("generated_games", None)
    study["wall_seconds"] = games_wall
    study["notebook_seconds"] = games_wall + 1800
    study["competition"] = {**base["competition"], "preview_instances": len(games), "preview_seconds": games_wall,
                            "preview_live_score": False}
    loop_cfg = base.get("rsi2_loop") or base.get("rsi2_worker_of") or {}  # a worker VM's bundle keeps it aside
    study["trials"] = [loop_cfg.get("eval_trial") or {"name": "duck-transfer", "kind": "duck-transfer"}]
    for key in ("rsi2_loop", "rsi2_worker", "rsi2_worker_of", "nebius_exchange_in"):
        study.pop(key, None)
    study.update(extra or {})
    (run / "configs/comparison.json").write_text(json.dumps(study, indent=1))
    for name in ("pi", "pi-arena"):  # node agents some harness paths use (read-only, gen 0's)
        if (bundle / name).is_dir() and not (run / name).exists():
            os.symlink(bundle / name, run / name)
    subprocess.run(["chmod", "-R", "a+rX,a-w", str(run)], check=True)
    return study


class Run:
    """One arena run: gateway + sandboxed ``arena.run``; ``marks`` {name: seconds} are watched from the snapshots."""

    def __init__(self, loop, cid, tree, *, kind, games, games_dir, wall, marks=None, tag=None, prefixes=None,
                 log_actions=False):
        self.loop, self.cid, self.tree, self.kind = loop, cid, Path(tree), kind
        self.prefixes, self.log_actions = prefixes, log_actions  # frontier: {exposed id: spec}; pool runs: True
        self.games, self.games_dir, self.wall, self.marks = list(games), str(games_dir), float(wall), marks or {}
        self.id = tag or loop.new_id("e")
        self.dir = loop.root / "eval" / self.id
        self.out = self.dir / "out"
        self.snapshots = self.dir / "snapshots.jsonl"
        self.actions = self.dir / "actions.jsonl"
        self.result = {"id": self.id, "candidate": cid, "kind": kind, "games": len(self.games), "wall": self.wall,
                       "marks": {}, "status": "running"}

    def argv(self, run, port):
        venv = Path(sys.prefix)  # the agent venv (never resolved: its python is a link to the base interpreter)
        env = {"HOME": "/tmp/home", "USER": EVAL_USER, "LANG": "C.UTF-8", "PYTHONPATH": "/opt/candidate/src",
               "PYTHONNOUSERSITE": "1", "PYTHONDONTWRITEBYTECODE": "1", "HF_HUB_OFFLINE": "1",
               "TRANSFORMERS_OFFLINE": "1", "PIP_NO_INDEX": "1", "MPLBACKEND": "Agg", "ONLY_RESET_LEVELS": "true",
               "RSI2_GATEWAY_URL": f"http://127.0.0.1:{port}",
               "PATH": f"{venv}/bin:/usr/local/bin:/usr/bin:/bin"}
        server_dir = Path(self.loop.stacks.server_dir)
        args = [str(venv / "bin/python"), "-m", "arena.run", "--source", "/opt/candidate", "--study",
                "/opt/candidate/configs/comparison.json", "--output", "/out", "--trial", "duck-transfer",
                "--server-dir", "/opt/server", "--base-url", self.loop.upstream,
                "--notebook-start-epoch", str(time.time()), "--games-dir", "/nonexistent-games"]
        tmp = self.dir / "tmp"
        (tmp / "home").mkdir(parents=True, exist_ok=True)
        sandbox.own(tmp, EVAL_USER, 0o700)
        sandbox.own(tmp / "home", EVAL_USER, 0o700)
        binds = [(run, "/opt/candidate", True), (self.out, "/out", False), (server_dir, "/opt/server", True)]
        tmp_binds = []
        if str(venv).startswith("/tmp/"):  # the agent venv lives under the host /tmp, covered by the private one
            tmp_binds.append((venv, str(venv).removeprefix("/tmp/"), True))
        hide = [p for p in self.loop.runtime.hidden() if p not in ("/kaggle",)]
        return sandbox.command(args, EVAL_USER, None, env, cwd="/out", extra_binds=binds, hide=hide, tmp=tmp,
                               tmp_binds=tmp_binds, shm_size="16g", net="host",
                               memory=self._memory())

    def _memory(self):
        total = self.loop.runtime.memory("eval")
        return int(total * getattr(self, "mem_share", 1.0)) if total else total

    def scores_at(self, seconds, started, clears=2):
        """Per source game score / levels from the last snapshot at or before ``started + seconds``, and the continuous
        signal: ``cont`` = 100 x mean over the first ``clears`` level clears of (1 - min(T, M) / M),
        T = seconds from ``started`` to the first snapshot showing that clear, M = ``seconds`` (censored: 0 if not
        cleared by the mark). Everything from the gateway's own scorecard snapshots."""
        best, first = None, {}
        if self.snapshots.exists():
            for line in self.snapshots.read_text().splitlines():
                try:
                    snap = json.loads(line)
                except ValueError:
                    continue
                if snap["t"] > started + seconds + 1:
                    break
                best = snap
                if not snap["cards"]:
                    continue
                card = max(snap["cards"].values(), key=lambda c: len(c["games"]))
                for exposed, row in card["games"].items():
                    source = snap["source"].get(exposed, exposed)
                    for k in range(1, clears + 1):
                        if (row.get("levels") or 0) >= k:
                            first.setdefault((source, k), snap["t"] - started)
        if best is None or not best["cards"]:
            return None
        card = max(best["cards"].values(), key=lambda c: len(c["games"]))
        out = {}
        for exposed, row in card["games"].items():
            source = best["source"].get(exposed, exposed)
            out[source] = {"score": float(row["score"] or 0.0), "levels": row["levels"]}
        for game in self.games:  # a game not yet started scores 0
            out.setdefault(game, {"score": 0.0, "levels": 0})
            out[game]["cont"] = round(100 * sum(1 - min(first.get((game, k), 1e18), seconds) / seconds
                                                for k in range(1, clears + 1)) / clears, 3)
        return {"t": best["t"], "games": out}

    def first_clears(self, targets, started, wall):
        """Frontier attempts: {exposed id: seconds from ``started`` to the first snapshot with levels >= target}
        (None if not by ``wall``), and the void instances (a replay that did not reproduce its board)."""
        first, void = {g: None for g in targets}, set()
        if self.snapshots.exists():
            for line in self.snapshots.read_text().splitlines():
                try:
                    snap = json.loads(line)
                except ValueError:
                    continue
                void |= set(snap.get("void") or [])
                if snap["t"] > started + wall + 1 or not snap["cards"]:
                    continue
                card = max(snap["cards"].values(), key=lambda c: len(c["games"]))
                for exposed, row in card["games"].items():
                    if exposed in first and first[exposed] is None and (row.get("levels") or 0) >= targets[exposed]:
                        first[exposed] = snap["t"] - started
        return first, void

    def stop_request(self, why):
        folder = self.out / "duck-transfer"
        folder.mkdir(parents=True, exist_ok=True)
        (folder / "stop-request").write_text(json.dumps({"reason": why, "by": "rsi2 racing"}))

    async def go(self, decide=None):
        """Run to the end (or a cut); ``decide(mark, scores) -> "continue" | "cut"`` at each mark."""
        load0 = proposer_load(self.loop)
        self.dir.mkdir(parents=True, exist_ok=True)
        os.chmod(self.dir, 0o700)
        run = self.dir / "source"
        compose(self.loop.source, self.tree, run, games=self.games, games_wall=self.wall)
        self.out.mkdir(parents=True, exist_ok=True)
        sandbox.own(self.out, EVAL_USER, 0o700)
        if self.prefixes:
            (self.dir / "prefixes.json").write_text(json.dumps(self.prefixes))
            os.chmod(self.dir / "prefixes.json", 0o600)
        port = free_port()
        self.loop.runtime.allow_gateway(port)
        log = (self.dir / "gateway.log").open("ab")
        gateway = await asyncio.create_subprocess_exec(
            sys.executable, "-m", "rsi2_loop.gateway", "--games-dir", self.games_dir, "--games", ",".join(self.games),
            "--port", str(port), "--snapshots", str(self.snapshots),
            *(["--log-actions", str(self.actions)] if self.log_actions else []),
            *(["--prefixes", str(self.dir / "prefixes.json")] if self.prefixes else []),
            env={**os.environ, "PYTHONPATH": str(self.loop.runtime.lib)}, stdout=log, stderr=subprocess.STDOUT,
            start_new_session=True)
        try:
            for _ in range(120):
                if (self.dir / "gateway.log").read_text(errors="replace").count("rsi2_gateway_ready"):
                    break
                if gateway.returncode is not None:
                    raise RuntimeError("gateway exited: " + (self.dir / "gateway.log").read_text()[-800:])
                await asyncio.sleep(1)
            else:
                raise RuntimeError("gateway not ready in 120 s")
            proxy = self.loop.proxy  # arena.run starts only on an idle server: the proposer's requests wait
            proxy.holds = getattr(proxy, "holds", 0) + 1  # practice runs start together: the last one reopens
            proxy.open.clear()
            try:
                held = time.time()
                while sum(proxy.inflight.values()) > 0 and time.time() - held < 900:
                    await asyncio.sleep(2)
                started = time.time()
                self.result["started"] = started
                with (self.dir / "arena.log").open("ab") as stream:
                    process = await asyncio.create_subprocess_exec(*self.argv(run, port), stdout=stream,
                                                                   stderr=subprocess.STDOUT, start_new_session=True)
                while process.returncode is None and time.time() - started < 300:
                    if b"starting_trial" in (self.dir / "arena.log").read_bytes():
                        break
                    await asyncio.sleep(2)
                started = time.time()  # the marks count from the trial's start (as progress elapsed_seconds)
                self.result["started"] = started
            finally:
                proxy.holds -= 1
                if proxy.holds <= 0:
                    proxy.open.set()
            pending = dict(sorted(self.marks.items(), key=lambda kv: kv[1]))
            cut = None
            hard_end = started + self.wall + 1800
            while process.returncode is None:
                await asyncio.sleep(15)
                elapsed = time.time() - started
                for name, seconds in list(pending.items()):
                    if elapsed >= seconds + 30:  # one more snapshot after the mark
                        scores = self.scores_at(seconds, started)
                        verdict = decide(name, scores) if decide else "continue"
                        self.result["marks"][name] = {"seconds": seconds, "verdict": verdict,
                                                      "games": (scores or {}).get("games")}
                        pending.pop(name)
                        self.loop.event_log({"event": "rsi2_mark", "run": self.id, "candidate": self.cid,
                                             "mark": name, "verdict": verdict})
                        self.loop.save("mark")
                        if verdict == "cut" and cut is None:
                            cut = name
                            self.stop_request(f"cut at {name}")
                if time.time() > hard_end:
                    os.killpg(process.pid, signal.SIGKILL)
                    self.result["status"] = "killed (hard end)"
                    break
                if self.loop.stop_requested() and not cut:
                    self.stop_request("loop stop")
            await process.wait()
            elapsed = time.time() - started
            for name, seconds in pending.items():  # marks after the end: a run that finished every game early
                if not cut and (elapsed >= seconds or process.returncode == 0):  # scores stay as they ended
                    scores = self.scores_at(seconds, started)
                    self.result["marks"][name] = {"seconds": seconds, "verdict": decide(name, scores) if decide
                                                  else "continue", "games": (scores or {}).get("games")}
            self.result.update(ended=time.time(), exit=process.returncode, cut=cut, seconds=round(elapsed))
            if process.returncode != 0:
                self.result["log_tail"] = (self.dir / "arena.log").read_bytes()[-3000:].decode(errors="replace")
            if self.result["status"] == "running":
                self.result["status"] = "complete" if process.returncode == 0 else f"exit {process.returncode}"
        finally:
            if gateway.returncode is None:
                gateway.send_signal(signal.SIGTERM)
                try:
                    await asyncio.wait_for(gateway.wait(), 60)
                except asyncio.TimeoutError:
                    os.killpg(gateway.pid, signal.SIGKILL)
            log.close()
            subprocess.run(["pkill", "-KILL", "-u", EVAL_USER], check=False)
            keep = self.loop.ckpt / "evals" / self.id  # root-side logs for the operator (the output is synced)
            keep.mkdir(parents=True, exist_ok=True)
            for name in ("arena.log", "gateway.log", "snapshots.jsonl", "actions.jsonl"):
                if (self.dir / name).is_file():
                    shutil.copyfile(self.dir / name, keep / name)
            for name in ("duck-transfer-process.log", "study.json", "duck-transfer/execution.json",
                         "duck-transfer/exception.txt", "duck-transfer/progress.jsonl"):
                try:
                    from .candidates import copy_regular
                    (keep / name).parent.mkdir(parents=True, exist_ok=True)
                    copy_regular(self.out / name, keep / name)
                except (OSError, ValueError):
                    pass
            self.loop.runtime.forbid_gateway(port)
            load1 = proposer_load(self.loop)
            if load0 is not None and load1 is not None:  # the proposer shares this GPU: its load during the run
                self.result["proposer_load"] = {k: load1[k] - load0[k] for k in load1}
        return self.result
