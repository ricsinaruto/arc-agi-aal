"""dfranzen's public notebook with his own serving stack on a Nebius RTX PRO 6000 VM.

The Nebius counterpart of scripts/modal_dfranzen_exact.py ``run``; the bundle's entry (scripts/nebius_runner.py calls it
with --bundle --working --output --games-dir --max-seconds, in a loopback-only network namespace, /kaggle/input ->
/assets/flash-input). His notebook's code cells are byte-identical (sha256 checked); the job adds ONE cell in front of
his ``await bm.run(...)`` cell (``STOP_CELL``: a stop request calls his own graceful-stop hook, ``bm.request_stop()``).
Everything else is plumbing around the notebook:

- a kernel venv from the bundle's wheels (IPython 8.37 / ipykernel 6.29.5 as on the Modal run; the scientific stack at
  the versions of b2d-24h-v2's harness venv), the notebook run top to bottom by nbclient in /kaggle/working with
  ``KAGGLE_IS_COMPETITION_RERUN=1`` (his competition branch) and ``ARC_BASE_URL`` at the local gateway;
- the 110-clone competition gateway of b2d-24h-v2 (``gateway.py`` = its b200_gateway.py, arc_agi 0.9.9 in its own venv,
  the games folder and ONLY_RESET_LEVELS=true of that run, clones k000..k109 of the study roster in order), port 8002;
- sidecars the notebook never reads: SGLang ``/metrics`` every 15 s, ``<output>/live/progress.json`` + ``.jsonl`` every
  minute in the arena's live shape (gateway score, family-balanced score, generated tokens), a mark file with the
  full scorecard when the generated tokens pass each ``marks`` value;
- the stop rule: generated tokens since the first game request >= ``stop_generated_tokens``, or ``play_backstop_seconds``
  of play (before his scheduler's tail fade at 80% of his wall), or a stop request (ARENA_STOP_REQUEST). The stop
  touches DFR_STOP_FILE; his teardown runs; after ``stop_grace_seconds`` the notebook is killed.

At the end ``<output>/working`` holds his /kaggle/working (per-instance ``artifacts/*_events.jsonl``: the per-action
history, transcripts, serve.log), ``<output>/gateway`` the scorecard series, ``<output>/dfr-result.json`` the summary.
``--selftest`` (no GPU): venvs, gateway, the notebook's CPU cells, the stop hook against his Benchmark class, the
progress writer and the stop rule against a fake metrics endpoint.
"""

import argparse
import hashlib
import json
import os
import re
import shutil
import signal
import subprocess
import sys
import threading
import time
import urllib.request
from pathlib import Path

HERE = Path(__file__).resolve().parent
NOTEBOOK = HERE / "notebook/arc-agi-3-milestone-2-solution.json"  # the packers drop *.ipynb
GATEWAY_PORT = 8002
RUN_CELL_MARK = "await bm.run(soft_end_time=soft_end, runtime_environment=target, minimal_diagnostics=TRUE_SUBMISSION)"
KERNEL_REQUIREMENTS = ["numpy", "scipy", "matplotlib", "pillow", "imageio", "requests", "pandas", "pyarrow",
                       "python-dotenv", "ipython==8.37.0", "ipykernel==6.29.5", "nbclient", "nbformat"]
# Set by the runner for arena bundles; not part of his notebook's environment on Kaggle or on the Modal run.
RUNNER_ENV = ("PYTHONPATH", "PYTHONNOUSERSITE", "HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE", "PIP_NO_INDEX",
              "ONLY_RESET_LEVELS", "PYTHONDONTWRITEBYTECODE", "ARC_VALIDATION_KIND", "ARC_VALIDATION_TEST",
              "ARC_VALIDATION_OUTPUT", "ARC_FORCE_CHROOT", "ARENA_STOP_REQUEST", "LIBRARY_PATH")
STOP_CELL = '''\
# Plumbing cell added by the Nebius job (not dfranzen's code): a stop request calls his graceful-stop hook.
import asyncio as _arc_asyncio, os as _arc_os

async def _arc_stop_watch(_path=_arc_os.environ.get("DFR_STOP_FILE", "")):
    while _path and not _arc_os.path.exists(_path):
        await _arc_asyncio.sleep(5)
    while _path and getattr(bm, "_solver_task", None) is None:
        await _arc_asyncio.sleep(1)  # request_stop() is a no-op before run() awaits the solver
    if _path:
        bm.request_stop()
        open(_path + ".called", "w").write("bm.request_stop()")  # a print from this task reaches no cell output

_arc_stop_task = _arc_asyncio.ensure_future(_arc_stop_watch())
print("[arc] stop watcher armed:", _arc_os.environ.get("DFR_STOP_FILE"), flush=True)
'''
EQUAL_CELL = '''\
# Plumbing (not dfranzen's code): the equal scheduler in place of his priority gate (dfr_equal.py).
import importlib.util as _arc_iu
_arc_spec = _arc_iu.spec_from_file_location("dfr_equal", _arc_os.environ["DFR_EQUAL_PY"])
_arc_equal = _arc_iu.module_from_spec(_arc_spec)
_arc_spec.loader.exec_module(_arc_equal)
_arc_equal_gate = _arc_equal.install()
'''
SELFTEST_CELL = '''\
import arc_agi, taaf.game_api, asyncio, threading, importlib
spec = taaf.game_api.ArcadeSpec(operation_mode=arc_agi.OperationMode.COMPETITION,
                                arc_base_url=os.environ['ARC_BASE_URL'], environments_dir='')
arcade = arc_agi.Arcade(operation_mode=arc_agi.OperationMode.COMPETITION, arc_base_url=spec.arc_base_url,
                        environments_dir='')
ids = [e.game_id for e in arcade.available_environments]
print('GATEWAY_GAMES', len(ids), ids[:3], 'client arc_agi', getattr(arc_agi, '__version__', None))
print('SOLVER', bm.solver.concurrency, bm.solver.max_runtime_s_per_game, bm.solver.save_request_logs, bm.n_passes,
      TRUE_SUBMISSION, os.environ.get('ARC3_MAX_ACTIVE_STREAMS'), os.environ.get('ARC3_PRIORITY_TAIL_FADE_FRACTION'))
[importlib.import_module(m) for m in ('inference.framework.solver', 'inference.agent.tool_agent',
                                     'inference.agent.priority_scheduler', 'inference.utils.frame_diff')]
print('IMPORTS_OK')
card = arcade.open_scorecard()
env = arcade.make('k000', scorecard_id=card)
print('MAKE_OK', type(env).__name__)
# The stop hook against his Benchmark.request_stop: the watcher must cancel the task run() would be awaiting.
bm._solver_task = asyncio.ensure_future(asyncio.sleep(600))
threading.Timer(3.0, lambda: open(os.environ['DFR_STOP_FILE'], 'w').write('selftest')).start()
try:
    await bm._solver_task
    print('STOP_HOOK_FAILED')
except asyncio.CancelledError:
    print('STOP_HOOK_OK')
'''


SELFTEST_PUBLIC_CELL = '''\
import arc_agi, taaf.game_api, asyncio, threading, importlib, numpy as np, imageio.v3 as iio
print('SOLVER', TRUE_SUBMISSION, bm.n_passes, bm.solver.concurrency, bm.solver.max_runtime_s_per_game,
      bm.solver.save_request_logs, target.max_runtime_s, demo_excluded_games,
      os.environ.get('ARC3_MAX_ACTIVE_STREAMS'), os.environ.get('ARC3_PRIORITY_TAIL_FADE_FRACTION'))
if (TRUE_SUBMISSION, bm.n_passes, bm.solver.concurrency, bm.solver.max_runtime_s_per_game) == (False, 4, 80, 23214):
    print('SOLVER_OK')
env_dir = str(Path("/kaggle/input/competitions/arc-prize-2026-arc-agi-3/arc_agi_3_wheels").parent / "environment_files")
spec = taaf.game_api.ArcadeSpec(operation_mode=arc_agi.OperationMode.OFFLINE, environments_dir=env_dir)
arcade = arc_agi.Arcade(operation_mode=arc_agi.OperationMode.OFFLINE, environments_dir=env_dir)
ids = [e.game_id for e in arcade.available_environments]
ids = [g for g in ids if not any(g.startswith(x) for x in demo_excluded_games)]
print('GAMES_OK', len(ids), ids, 'arc_agi', getattr(arc_agi, '__version__', None))
[importlib.import_module(m) for m in ('inference.framework.solver', 'inference.agent.tool_agent',
                                     'inference.agent.priority_scheduler', 'inference.utils.frame_diff',
                                     'taaf.diagnostics')]
print('IMPORTS_OK')
env = arcade.make(ids[0])
print('MAKE_OK', type(env).__name__)
iio.imwrite('/tmp/arc-selftest.mp4', np.zeros((4, 128, 128, 3), dtype=np.uint8), fps=4, codec='libx264',
            macro_block_size=1, output_params=['-movflags', '+faststart'])
print('MP4_OK', os.path.getsize('/tmp/arc-selftest.mp4'))
bm._solver_task = asyncio.ensure_future(asyncio.sleep(600))
threading.Timer(3.0, lambda: open(os.environ['DFR_STOP_FILE'], 'w').write('selftest')).start()
try:
    await bm._solver_task
    print('STOP_HOOK_FAILED')
except asyncio.CancelledError:
    print('STOP_HOOK_OK')
'''


def log(event, **kw):
    print(json.dumps({"event": event, **kw}), flush=True)


def sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        while block := stream.read(8 << 20):
            digest.update(block)
    return digest.hexdigest()


def make_venv(path, wheels, requirements):
    """A venv of the VM's python3.12 with ``requirements`` from the bundle's wheels only."""
    subprocess.run([sys.executable, "-m", "venv", str(path)], check=True)
    subprocess.run([str(path / "bin/python"), "-m", "pip", "install", "--quiet", "--no-index", "--find-links",
                    str(wheels), "--disable-pip-version-check", *requirements], check=True,
                   env={**os.environ, "PIP_CONFIG_FILE": os.devnull})
    return path / "bin/python"


def prepare_notebook(target, expected_sha, notebook=None, edits=(), extra=""):
    """His notebook with the stop cell in front of his run cell; his cells are left as they are, apart from the
    study's listed line edits (``{"cell", "old", "new"}``, each ``old`` exactly once in its cell)."""
    raw = Path(notebook or NOTEBOOK).read_bytes()
    if hashlib.sha256(raw).hexdigest() != expected_sha:
        raise RuntimeError("notebook hash mismatch")
    nb = json.loads(raw)
    text = lambda c: c["source"] if isinstance(c["source"], str) else "".join(c["source"])  # noqa: E731
    hits = [i for i, c in enumerate(nb["cells"]) if c["cell_type"] == "code" and RUN_CELL_MARK in text(c)]
    if len(hits) != 1:
        raise RuntimeError(f"expected one run cell, found {hits}")
    before = [hashlib.sha256(text(c).encode()).hexdigest() for c in nb["cells"]]
    for edit in edits:
        source = text(nb["cells"][edit["cell"]])
        if source.count(edit["old"]) != 1 or edit["old"] not in source.splitlines():
            raise RuntimeError(f"edit does not match one whole line of cell {edit['cell']}")
        nb["cells"][edit["cell"]]["source"] = source.replace(edit["old"], edit["new"]).splitlines(True)
        before[edit["cell"]] = hashlib.sha256(text(nb["cells"][edit["cell"]]).encode()).hexdigest()
    nb["cells"].insert(hits[0], {"cell_type": "code", "execution_count": None, "outputs": [],
                                 "metadata": {"arc_plumbing": "stop watcher (not dfranzen's)"
                                                               + (" + equal scheduler" if extra else "")},
                                 "source": (STOP_CELL + extra).splitlines(True)})
    after = [hashlib.sha256(text(c).encode()).hexdigest() for i, c in enumerate(nb["cells"]) if i != hits[0]]
    if before != after:
        raise RuntimeError("a notebook cell changed")
    Path(target).write_text(json.dumps(nb, indent=1))
    return {"inserted_cell_index": hits[0], "run_cell_index": hits[0] + 1, "cells": len(nb["cells"]),
            "his_cells_unchanged": not edits, "edits": list(edits), "notebook_sha256": expected_sha}


def kernel_env(venv_python, stop_file, public=False):
    env = {k: v for k, v in os.environ.items() if k not in RUNNER_ENV}
    if not public:  # his competition branch against the local gateway; his public run sets neither
        env.update(KAGGLE_IS_COMPETITION_RERUN="1", ARC_BASE_URL=f"http://127.0.0.1:{GATEWAY_PORT}/")
    env.update(PYTHONUNBUFFERED="1", MPLBACKEND="Agg", DFR_STOP_FILE=str(stop_file),
               VIRTUAL_ENV=str(Path(venv_python).parents[1]),
               PATH=f"{Path(venv_python).parent}:{os.environ.get('PATH', '/usr/local/bin:/usr/bin:/bin')}")
    return env


def start_gateway(python, games_dir, roster, instances, out, log_file):
    proc = subprocess.Popen(
        [str(python), "-u", str(HERE / "gateway.py"), "--environments-dir", str(games_dir), "--games",
         json.dumps(roster), "--total", str(instances), "--port", str(GATEWAY_PORT), "--out", str(out),
         "--interval", "30"], stdout=log_file, stderr=subprocess.STDOUT, start_new_session=True,
        env={**{k: v for k, v in os.environ.items() if k != "PYTHONPATH"}, "ONLY_RESET_LEVELS": "true",
             "RECORDINGS_DIR": "/tmp/gateway-recordings"})
    for _ in range(180):
        try:
            with urllib.request.urlopen(f"http://127.0.0.1:{GATEWAY_PORT}/api/games", timeout=2) as r:
                if r.status == 200 and len(json.loads(r.read())) == instances:
                    return proc
        except Exception:  # noqa: BLE001 - not up yet
            pass
        if proc.poll() is not None:
            raise RuntimeError("gateway exited")
        time.sleep(1)
    raise RuntimeError("gateway not ready")


def game_content(path):
    """A game file without what differs between two downloads of one version: the licence comment block at the top
    of the source, ``local_dir`` and ``date_downloaded`` in the metadata."""
    text = Path(path).read_text()
    if path.name == "metadata.json":
        return {k: v for k, v in json.loads(text).items() if k not in ("local_dir", "date_downloaded")}
    lines = text.splitlines()
    while lines and (lines[0].startswith("#") or not lines[0].strip()):
        lines.pop(0)
    return lines


def games_match(games_dir, competition_dir, roster):
    """The roster's environment files in the b2d games folder against his competition dataset's."""
    rows = {}
    for game in roster:
        name, version = game.split("-")
        pair = [sorted(p for p in (Path(root) / name / version).glob("*") if p.is_file())
                for root in (games_dir, competition_dir)]
        rows[game] = ([p.name for p in pair[0]] == [p.name for p in pair[1]] and bool(pair[0])
                      and all(game_content(a) == game_content(b) for a, b in zip(*pair)))
    return rows


def scrape(url):
    """SGLang's Prometheus samples (no buckets), summed over label sets, or None."""
    try:
        with urllib.request.urlopen(url, timeout=5) as r:
            text = r.read().decode()
    except Exception:  # noqa: BLE001 - monitoring only
        return None
    sample = {}
    for line in text.splitlines():
        if line.startswith("#") or "_bucket{" in line:
            continue
        key, _, value = line.rpartition(" ")
        try:
            name = key.split("{", 1)[0]
            sample[name] = sample.get(name, 0.0) + float(value)
        except ValueError:
            pass
    return sample


def family_scores(row, sources):
    """``(family-balanced mean, {family: mean clone score})`` of one gateway scorecard row."""
    by = {}
    for env in row.get("environments") or []:
        by.setdefault(env.get("family") or sources.get(env["id"], env["id"]), []).append(float(env["score"]))
    families = {name: sum(v) / len(v) for name, v in sorted(by.items())}
    return (sum(families.values()) / len(families) if families else None), families


def benchmark_row(path):
    """His periodic ``benchmark.json`` (public mode, every 600 s) as a scorecard-like row: official score per
    instance (level = min(1.15, (baseline / actions)^2), index weights, capped at the weight of the cleared levels)."""
    data = json.loads(Path(path).read_text())
    envs, seen = [], {}
    for run in data["game_runs"]:
        n, done = run.get("number_of_levels") or 0, run.get("levels_completed") or 0
        base, used = run.get("base_actions_per_level") or [], run.get("actions_per_level") or []
        raw = sum((i + 1) * min(1.15, (base[i] / used[i]) ** 2) for i in range(min(done, len(base), len(used)))
                  if base[i] and used[i])
        score = min(raw, done * (done + 1) / 2) / (n * (n + 1) / 2) * 100 if n else 0.0
        k = seen[run["game_id"]] = seen.get(run["game_id"], -1) + 1
        envs.append({"id": f"{run['game_id']}_p{k}", "family": run["game_id"], "score": score, "levels": done,
                     "actions": len(run.get("history") or []), "state": run.get("state"),
                     "tokens": sum(h.get("generated_tokens") or 0 for h in run.get("history") or [])})
    return {"epoch": Path(path).stat().st_mtime, "score_mean_110": sum(e["score"] for e in envs) / max(1, len(envs)),
            "levels": sum(e["levels"] for e in envs), "actions": sum(e["actions"] for e in envs),
            "games_won": sum(e["state"] == "won" for e in envs),
            "games_played": sum(e["state"] != "not_started" for e in envs), "environments": envs}


class Monitor:
    """Live rows, token marks and the stop rule. ``tick`` runs every 15 s; nothing here is read by the notebook."""

    def __init__(self, output, block, metrics_url, stop_file, request_file, sources, clock=time.time):
        self.output, self.block, self.url = Path(output), block, metrics_url
        self.stop_file, self.request_file, self.sources, self.clock = Path(stop_file), request_file, sources, clock
        self.started = clock()
        self.play_started = self.tokens0 = self.reason = self.stopped_at = None
        self.samples, self.marks, self.last_row = [], {}, 0.0
        self.benchmark = self.benchmark_mtime = None
        (self.output / "live").mkdir(parents=True, exist_ok=True)

    def gateway_row(self):
        if self.block.get("mode") == "public":
            source = Path("/kaggle/working/benchmark.json")
            try:
                if source.stat().st_mtime != self.benchmark_mtime:
                    self.benchmark, self.benchmark_mtime = benchmark_row(source), source.stat().st_mtime
                    shutil.copyfile(source, self.output / "live/benchmark.json")
            except (OSError, ValueError, KeyError, TypeError, ZeroDivisionError):
                pass
            return self.benchmark
        try:
            lines = (self.output / "gateway/scorecards.jsonl").read_text().splitlines()
            rows = [json.loads(x) for x in lines[-3:]]
            rows = [r for r in rows if "error" not in r]
            return rows[-1] if rows else None
        except (OSError, ValueError):
            return None

    def tick(self):
        now = self.clock()
        sample = scrape(self.url)
        row = self.gateway_row()
        generated = None if sample is None else sample.get("sglang:generation_tokens_total")
        if sample is not None:
            with (self.output / "metrics.jsonl").open("a") as f:
                f.write(json.dumps({"epoch": now, "m": sample}) + "\n")
        if self.block.get("mode") == "public":  # no gateway: play starts with his run cell
            try:
                playing = "Starting benchmark..." in (self.output / "notebook.log").read_text(errors="replace")
            except OSError:
                playing = False
        else:
            playing = row is not None
        if self.play_started is None and playing and generated is not None:
            # The first scorecard row: his games have started; tokens before it (server warm-up) do not count.
            self.play_started, self.tokens0 = now, generated
            log("play_started", generated_before=generated)
        tokens = None if generated is None or self.tokens0 is None else generated - self.tokens0
        elapsed = None if self.play_started is None else now - self.play_started
        if tokens is not None:
            self.samples.append((elapsed, tokens))
            for mark in self.block.get("marks") or []:
                if tokens >= mark and mark not in self.marks and row is not None:
                    balanced, families = family_scores(row, self.sources)
                    self.marks[mark] = {"tokens": tokens, "elapsed_seconds": round(elapsed, 1), "epoch": now,
                                        "score_mean_110": row.get("score_mean_110"), "family_balanced": balanced,
                                        "levels": row.get("levels"), "actions": row.get("actions")}
                    (self.output / f"live/mark-{int(mark)}.json").write_text(json.dumps(
                        {**self.marks[mark], "families": families, "scorecard_row": row}, indent=1))
                    log("token_mark", mark=mark, **self.marks[mark])
        if self.reason is None:
            limit = self.block.get("stop_generated_tokens")
            if tokens is not None and limit and tokens >= limit:
                self.reason = f"generated tokens {tokens:.0f} >= {limit}"
            elif (self.block.get("stop_at_backstop", True) and elapsed is not None
                  and elapsed >= self.block["play_backstop_seconds"]):
                self.reason = f"play backstop {self.block['play_backstop_seconds']} s (before his tail fade)"
            elif self.request_file and Path(self.request_file).exists():
                self.reason = "stop_request: " + Path(self.request_file).read_text()[:200].strip()
            if self.reason:
                self.stopped_at = now
                self.stop_file.write_text(self.reason + "\n")
                log("stop", reason=self.reason, tokens=tokens, elapsed_seconds=elapsed)
        if now - self.last_row >= 60 or (self.reason and self.stopped_at == now):
            self.last_row = now
            self.write_row(now, row, sample or {}, tokens, elapsed)
        return self.reason

    def rate(self, window):
        if len(self.samples) < 2:
            return None
        end = self.samples[-1]
        old = [s for s in self.samples if s[0] <= end[0] - window]
        first = old[-1] if old else self.samples[0]
        return (end[1] - first[1]) / (end[0] - first[0]) if end[0] > first[0] else None

    def write_row(self, now, row, sample, tokens, elapsed):
        row = row or {}
        balanced, _ = family_scores(row, self.sources)
        score = None if row.get("score_mean_110") is None else {
            "score": row["score_mean_110"], "family_balanced": balanced, "levels": row.get("levels"),
            "games_won": row.get("games_won"), "games_played": row.get("games_played"), "actions": row.get("actions"),
            "at_epoch": row.get("epoch"),
            "at_elapsed_seconds": None if self.play_started is None else round(row["epoch"] - self.play_started, 1)}
        if score is not None and score["at_elapsed_seconds"] is not None and score["at_elapsed_seconds"] < 0:
            score = None  # a benchmark.json from before play
        prompt, cached = sample.get("sglang:prompt_tokens_total"), sample.get("sglang:cached_tokens_total")
        progress = {"epoch": round(now, 1), "elapsed_seconds": round(elapsed or 0.0, 1),
                    "phase": "stopping" if self.reason else "notebook" if self.play_started else "setup",
                    "levels": row.get("levels"), "finished": row.get("games_won"), "stop_reason": self.reason,
                    "live": {"score": score, "engine": {
                        "generated_tokens": tokens, "generated_tokens_per_clone": None if tokens is None
                        else tokens / self.block["instances"], "tokens_per_second_mean": self.rate(10 ** 9),
                        "tokens_per_second_30m": self.rate(1800), "running": sample.get("sglang:num_running_reqs"),
                        "queued": sample.get("sglang:num_queue_reqs"),
                        "prefix_hit_total": cached / prompt if prompt and cached is not None else None}},
                    "marks": {str(int(k)): v for k, v in self.marks.items()}}
        (self.output / "live/progress.json.tmp").write_text(json.dumps(progress, indent=1))
        (self.output / "live/progress.json.tmp").replace(self.output / "live/progress.json")
        with (self.output / "live/progress.jsonl").open("a") as f:
            f.write(json.dumps(progress) + "\n")
        log("live_progress", elapsed_seconds=progress["elapsed_seconds"], tokens=tokens,
            score=None if score is None else score["score"], family_balanced=balanced, levels=row.get("levels"),
            tok_s_30m=progress["live"]["engine"]["tokens_per_second_30m"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bundle", default=str(HERE))
    ap.add_argument("--working", default="/tmp/arena")
    ap.add_argument("--output", required=True)
    ap.add_argument("--games-dir", required=True)
    ap.add_argument("--max-seconds", type=int, required=True)
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--metrics-url", default="http://127.0.0.1:8001/metrics", help="the self-test's fake endpoint")
    args = ap.parse_args()
    started = time.time()
    bundle, working, output = Path(args.bundle).resolve(), Path(args.working), Path(args.output)
    study = json.loads((bundle / "configs/comparison.json").read_text())
    block, roster = study["dfr_exact"], study["games"]
    for folder in (working, output, Path("/kaggle/working")):
        folder.mkdir(parents=True, exist_ok=True)
    result = {"run": study["name"], "started_epoch": started, "selftest": args.selftest, "python": sys.version.split()[0]}
    procs, stop_file = {}, working / "dfr-stop-request"
    stop_file.unlink(missing_ok=True)
    competition = Path("/kaggle/input/competitions/arc-prize-2026-arc-agi-3")
    sources = {f"k{i:03d}": roster[i % len(roster)] for i in range(block["instances"])}
    public = block.get("mode") == "public"  # his public run: offline games in his kernel, no gateway, his own wall
    monitor = None
    try:
        missing = [p for p in block["assets"] if not (Path("/assets") / p).is_dir()]
        if missing:
            raise RuntimeError(f"inputs not on the machine: {missing}")
        if not public:
            same = games_match(args.games_dir, competition / "environment_files", roster)
            result["roster_games_equal_to_his_competition_files"] = all(same.values())
            result["roster_games_differing"] = [g for g, ok in same.items() if not ok]
        t0 = time.time()
        if not public:
            gateway_python = make_venv(working / "gateway-venv", bundle / "wheels", ["arc-agi==0.9.9"])
        kernel_python = make_venv(working / "kernel-venv", bundle / "wheels",
                                  KERNEL_REQUIREMENTS + list(block.get("kernel_extra") or []))
        result["venv_seconds"] = round(time.time() - t0, 1)
        result["kernel_packages"] = subprocess.run(
            [str(kernel_python), "-m", "pip", "list", "--format=freeze", "--disable-pip-version-check"],
            capture_output=True, text=True).stdout.split()
        if not public:
            procs["gateway"] = start_gateway(gateway_python, args.games_dir, roster, block["instances"],
                                             output / "gateway", (output / "gateway.log").open("ab"))
            result["gateway_ready_seconds"] = round(time.time() - started, 1)
        equal = block.get("equal")
        result["notebook"] = prepare_notebook(
            output / "notebook-input.ipynb", block["notebook_sha256"],
            HERE / block["notebook_file"] if block.get("notebook_file") else None, block.get("edits") or (),
            EQUAL_CELL if equal else "")
        env = kernel_env(kernel_python, stop_file, public)
        if equal:  # equal scheduler, swap log, per-request instance log
            (output / "live").mkdir(parents=True, exist_ok=True)
            config = output / "equal-config.json"
            config.write_text(json.dumps({"park_tokens": int(equal["park_tokens"]),
                                          "log_path": str(output / "equal-swaps.jsonl"),
                                          "state_path": str(output / "live/equal-state.json"),
                                          "requests_path": str(output / "requests.jsonl")}, indent=1))
            env.update(DFR_EQUAL_CONFIG=str(config), DFR_EQUAL_PY=str(HERE / "dfr_equal.py"))
        command = [str(kernel_python), "-u", str(HERE / "run_notebook.py"), str(output / "notebook-input.ipynb"),
                   str(output / "notebook-executed.ipynb")]
        if args.selftest:
            stop_index = result["notebook"]["inserted_cell_index"]
            command += ["--cells", f"2,4,8,10,14,16,{stop_index}", "--extra",
                        SELFTEST_PUBLIC_CELL if public else SELFTEST_CELL]
        monitor = Monitor(output, block, args.metrics_url, stop_file, os.environ.get("ARENA_STOP_REQUEST"), sources)
        log("notebook_start", command=command[-2:], seconds=round(time.time() - started, 1))
        nb_log = (output / "notebook.log").open("wb")
        procs["notebook"] = subprocess.Popen(command, env=env, cwd="/kaggle/working", stdout=subprocess.PIPE,
                                             stderr=subprocess.STDOUT, start_new_session=True)

        def pump():
            for line in iter(procs["notebook"].stdout.readline, b""):
                nb_log.write(line)
                nb_log.flush()
                sys.stdout.buffer.write(line)
                sys.stdout.flush()

        threading.Thread(target=pump, daemon=True).start()
        hard_end = started + args.max_seconds - 240  # room for the copy of his working folder
        last_copy = 0.0
        while procs["notebook"].poll() is None:
            time.sleep(1 if args.selftest else 15)
            reason = None if args.selftest else monitor.tick()
            if time.time() - last_copy > 300 and Path("/kaggle/working/serve.log").exists():
                last_copy = time.time()
                shutil.copyfile("/kaggle/working/serve.log", output / "serve.log")
            late = reason and time.time() - monitor.stopped_at > block["stop_grace_seconds"]
            if late or time.time() > hard_end:
                result["hard_stop"] = "his teardown did not end in the grace time" if late else "job time limit"
                os.killpg(procs["notebook"].pid, signal.SIGTERM)
                try:
                    procs["notebook"].wait(60)
                except subprocess.TimeoutExpired:
                    os.killpg(procs["notebook"].pid, signal.SIGKILL)
                break
        result["notebook_exit"] = procs["notebook"].wait()
        result["notebook_end_seconds"] = round(time.time() - started, 1)
        if not args.selftest:
            time.sleep(35)  # one more gateway snapshot after his teardown
            monitor.tick()
    except BaseException as exc:
        result["error"] = f"{type(exc).__name__}: {exc}"
        raise
    finally:
        for proc in procs.values():
            if proc.poll() is None:
                try:
                    os.killpg(proc.pid, signal.SIGKILL)
                except OSError:
                    pass
        subprocess.run(["pkill", "-9", "-f", "sglang"], check=False)
        kaggle_working = Path("/kaggle/working")
        if kaggle_working.exists():
            shutil.copytree(kaggle_working, output / "working", dirs_exist_ok=True,
                            ignore=shutil.ignore_patterns("*.pkl", "server_recording"))
        text = (output / "notebook.log").read_text(errors="replace") if (output / "notebook.log").exists() else ""
        ready = re.search(r"READY after (\d+)s", text)
        result["server_ready_after_launch_seconds"] = int(ready.group(1)) if ready else None
        result["his_stop_hook_called"] = Path(f"{stop_file}.called").exists()
        if monitor is not None:
            row = monitor.gateway_row() or {}
            balanced, families = family_scores(row, sources)
            result.update(stop_reason=monitor.reason, marks={str(int(k)): v for k, v in monitor.marks.items()},
                          generated_tokens=monitor.samples[-1][1] if monitor.samples else None,
                          play_seconds=monitor.samples[-1][0] if monitor.samples else None,
                          final={"score_mean_110": row.get("score_mean_110"), "family_balanced": balanced,
                                 "levels": row.get("levels"), "actions": row.get("actions"), "families": families})
        result["seconds"] = round(time.time() - started, 1)
        (output / "dfr-result.json").write_text(json.dumps(result, indent=1))
        log("dfr_result", **{k: v for k, v in result.items() if k not in ("kernel_packages", "final")},
            final_score=(result.get("final") or {}).get("score_mean_110"))
    if args.selftest:
        text = (output / "notebook.log").read_text(errors="replace")
        names = (("SOLVER_OK", "GAMES_OK 20", "IMPORTS_OK", "MAKE_OK", "MP4_OK", "STOP_HOOK_OK") if public else
                 (f"GATEWAY_GAMES {block['instances']}", "IMPORTS_OK", "MAKE_OK", "STOP_HOOK_OK"))
        names += ('"event": "dfr_equal_gate"',) if block.get("equal") else ()
        checks = {name: name in text for name in names}
        checks["stop hook marker"] = bool(result["his_stop_hook_called"])
        log("selftest", checks=checks, notebook_exit=result["notebook_exit"])
        return 0 if all(checks.values()) and result["notebook_exit"] == 0 else 1
    # After a requested stop his run cell ends with CancelledError (request_stop re-raises after the teardown).
    return 0 if monitor.reason and result.get("his_stop_hook_called") else result["notebook_exit"]


if __name__ == "__main__":
    raise SystemExit(main())
