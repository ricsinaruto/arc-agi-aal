"""Training-metrics exporter: the train-metrics jsonl that tools/ops/train_dash.py renders and tools/ops/watch.py reads.

Row: {"t": epoch, "m": {key: number}, "alert": text (optional), "note": text (optional)}. Keys group by prefix
(infra.*, time.*, train.*, eval.*, play.*). ``alert`` is for watcher-grade alerts only; history markers go in ``note``.
Beside it, ``<run>.meta.json``: {"run", "arm", "status" (live|ended|stopped), "what", "cards", "continues", ...}.

In a container: ``MetricsWriter(output / "train-metrics.jsonl", commit=volume.commit, host=True, abort=fn)``: host=True
starts the process's host-memory guard at once (from step 0: RSS per process, pinned, /dev/shm, /tmp, PSI every 30 s;
``abort(reason)`` ends the run before thrash, see hostguard.py). On this machine (exporters
that turn a run's feed into rows): ``writer(run)``, ``meta(run, arm=...)``, ``link(new, old)``.
"""
import json
import os
import threading
import time
from pathlib import Path


def scratch_dir():
    """<ARC_OPS_SCRATCH>/train-metrics (tools/ops/paths.py reads the same variable from runs/ops/ops.env)."""
    root = os.environ.get("ARC_OPS_SCRATCH")
    if not root:
        env = Path(__file__).resolve().parents[3] / "runs/ops/ops.env"
        for line in env.read_text().splitlines() if env.is_file() else []:
            k, _, v = line.strip().removeprefix("export ").partition("=")
            if k == "ARC_OPS_SCRATCH":
                root = v.strip().strip('"')
    return Path(root or Path(__file__).resolve().parents[3] / "runs/ops") / "train-metrics"


def _num(v):
    return float(f"{float(v):.6g}")


class MetricsWriter:
    """Appends rows to one jsonl. Thread-safe. ``commit``: called at most every ``commit_s`` (a Modal volume's
    commit, so the watcher sees rows while the run lives). ``infra.cost_usd`` never steps back within a file."""

    def __init__(self, path, commit=None, commit_s=60.0, host=False, abort=None, host_interval_s=30.0):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.commit_fn, self.commit_s, self._last_commit = commit, commit_s, 0.0
        self.lock = threading.Lock()
        self._cost = None
        self.host = None
        if host:
            from .hostguard import HostGuard

            self.host = HostGuard(self, interval_s=host_interval_s, abort=abort).start()

    def _cost_floor(self):
        if self._cost is None:
            self._cost = 0.0
            for row in read_rows(self.path):
                self._cost = max(self._cost, float((row.get("m") or {}).get("infra.cost_usd") or 0))
        return self._cost

    def emit(self, m=None, t=None, alert=None, note=None):
        m = {k: _num(v) for k, v in (m or {}).items() if isinstance(v, (int, float)) and not isinstance(v, bool)}
        if not m and not alert and not note:
            return None
        with self.lock:
            if "infra.cost_usd" in m:
                m["infra.cost_usd"] = self._cost = max(self._cost_floor(), m["infra.cost_usd"])
            row = {"t": float(time.time() if t is None else t), "m": m}
            if alert:
                row["alert"] = str(alert)[:300]
            if note:
                row["note"] = str(note)[:300]
            with open(self.path, "a") as f:
                f.write(json.dumps(row) + "\n")
            if self.commit_fn and time.time() - self._last_commit >= self.commit_s:
                self._last_commit = time.time()
                try:
                    self.commit_fn()
                except Exception as e:  # noqa: BLE001 - a commit failure never ends the run
                    print(f"RUNOPS metrics commit failed: {e!r}", flush=True)
        return row

    def alert(self, text, **m):
        return self.emit(m, alert=text)


def read_rows(path):
    rows = []
    try:
        lines = Path(path).read_text().splitlines()
    except FileNotFoundError:
        return rows
    for line in lines:
        try:
            rows.append(json.loads(line))
        except ValueError:
            continue
    return rows


_WRITERS = {}


def writer(run, root=None):
    """The scratch writer for ``run`` (one per process)."""
    path = Path(root or scratch_dir()) / f"{run}.jsonl"
    if path not in _WRITERS:
        _WRITERS[path] = MetricsWriter(path)
    return _WRITERS[path]


def meta(run, arm=None, what=None, cards=None, status=None, continues=None, root=None, **extra):
    """Merges into <run>.meta.json; never drops "continues" (the earlier legs train_dash joins into one view).
    ``continues`` entries: a run id, or {"run", "until"} (the parent's rows up to the branch point)."""
    f = Path(root or scratch_dir()) / f"{run}.meta.json"
    f.parent.mkdir(parents=True, exist_ok=True)
    old = json.loads(f.read_text()) if f.is_file() else {}
    new = {**old, "run": run, "status": status or old.get("status") or "live"}
    for k, v in (("arm", arm), ("what", what), ("cards", None if cards is None else int(cards)), *extra.items()):
        if v is not None:
            new[k] = v
    if continues:
        have = list(old.get("continues") or [])
        names = {c["run"] if isinstance(c, dict) else c for c in have} | {run}
        new["continues"] = have + [c for c in continues if (c["run"] if isinstance(c, dict) else c) not in names]
    if new != old:
        tmp = f.with_name(f".{f.name}.tmp")
        tmp.write_text(json.dumps(new))
        os.replace(tmp, f)
    return new


def link(new, old, stage=None, root=None):
    """``new`` continues ``old`` (relaunch, restart, ceiling handoff): one dashboard view; the old leg is ended."""
    prev = meta(old, status="ended", root=root)
    keep = {k: prev[k] for k in ("arm", "what", "cards", "summary", "label") if prev.get(k)}
    cur_f = Path(root or scratch_dir()) / f"{new}.meta.json"
    cur = json.loads(cur_f.read_text()) if cur_f.is_file() else {}
    fill = {k: v for k, v in keep.items() if not cur.get(k)}
    return meta(new, continues=[old], root=root, **({"stage": stage} if stage else {}), **fill)


def stale(run, now=None, limit=1800, root=None):
    """Seconds since the run's last metric row when that exceeds ``limit`` and the meta says live, else None."""
    d = Path(root or scratch_dir())
    mf = d / f"{run}.meta.json"
    if mf.is_file() and json.loads(mf.read_text()).get("status") not in (None, "live"):
        return None
    now = time.time() if now is None else now
    ts = [float(r["t"]) for r in read_rows(d / f"{run}.jsonl") if r.get("m") and r.get("t", 0) <= now]
    last = max(ts) if ts else (mf.stat().st_mtime if mf.is_file() else now)
    return now - last if now - last > limit else None
