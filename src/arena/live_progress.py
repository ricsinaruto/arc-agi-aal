"""Live progress readable from outside a running arena trial (hero-v2, "Check intermediate score
every 30 minutes").

``study["live_progress"] = {"interval_seconds": 60, "score_seconds": 300}`` adds a ``live`` block to the trial's
``progress.json`` (written every 15 s): the current score (read from the open scorecard every ``score_seconds``),
levels per game, completion tokens and engine throughput. The bootstrap copies it every ``interval_seconds`` to the
run's output folder (``live/progress.json``, ``live/progress.jsonl``) and prints one ``live_progress`` line, so a
Modal run shows it in ``modal app logs`` and in the runs volume (committed every 120 s by scripts/modal_validation.py);
``scripts/arena/watch_live.py`` prints it every 30 minutes and exits when the run ends.

Monitoring only: nothing here reaches an agent or the scheduler, and the score comes from the scorecard the trial
already holds (the local research gateway, or the offline arcade); the hidden gateway is never read.
"""

import json
import time
from pathlib import Path

DEFAULTS = {"interval_seconds": 60, "score_seconds": 300}
ENGINE = {"generated_tokens": "vllm:generation_tokens_total", "running": "vllm:num_requests_running",
          "waiting": "vllm:num_requests_waiting", "kv_cache_fraction": "vllm:kv_cache_usage_perc",
          "preemptions": "vllm:num_preemptions_total"}
RATE_WINDOW_SECONDS = 60
HIT_WINDOW_SECONDS = 300
PREFIX_QUERIES = "vllm:prefix_cache_queries_total"
PREFIX_HITS = ("vllm:prefix_cache_hits_total", "vllm:external_prefix_cache_hits_total")  # GPU + CPU tier


def settings(spec):
    if spec in (None, False):
        return None
    cfg = {**DEFAULTS, **(spec if isinstance(spec, dict) else {})}
    if set(cfg) != set(DEFAULTS) or any(type(cfg[k]) is not int or not 10 <= cfg[k] <= 3600 for k in DEFAULTS):
        raise ValueError(f"live_progress takes {sorted(DEFAULTS)} (10..3600 s)")
    return cfg


def read_score(arc, card, roster):
    """``{"score": mean over the roster, "levels": total, "scored_games": n}`` from the open scorecard, or None."""
    scorecard = None
    peek = getattr(type(arc), "peek_score", None)
    if peek is not None:
        scorecard = arc.peek_score(card)
    else:
        from arc_agi import OperationMode

        if getattr(arc, "operation_mode", None) == OperationMode.OFFLINE:
            scorecard = arc.get_scorecard(card)
    if scorecard is None:
        return None
    scores = {env.id: float(env.score) for env in scorecard.environments}
    return {"score": sum(scores.values()) / max(1, roster), "levels": int(scorecard.total_levels_completed),
            "scored_games": sum(1 for v in scores.values() if v > 0),
            "top": sorted(((round(v, 3), k) for k, v in scores.items() if v > 0), reverse=True)[:8]}


def prefix_hit(base, now):
    """Token-weighted prefix hit between two counter samples (GPU + CPU tier); None if the counters are missing."""
    try:
        queries = now[PREFIX_QUERIES] - base[PREFIX_QUERIES]
        hits = sum(now.get(k, 0) - base.get(k, 0) for k in PREFIX_HITS)
        return round(hits / queries, 4) if queries > 0 else None
    except (KeyError, TypeError):
        return None


def engine_snapshot(samples, window=RATE_WINDOW_SECONDS):
    """Counters of the last engine sample and the generation rate over the last ``window`` s and since the start."""
    if not samples:
        return None
    last = samples[-1]
    metrics = last["metrics"]
    from arc_infra.metrics import generation_counter_metadata

    result = {label: metrics.get(key) for label, key in ENGINE.items()}
    result.update(generation_counter_metadata(metrics))
    key = ENGINE["generated_tokens"]
    first = samples[0]
    recent = next((s for s in reversed(samples) if last["elapsed_seconds"] - s["elapsed_seconds"] >= window), first)
    for label, base in (("tokens_per_second_recent", recent), ("tokens_per_second_mean", first)):
        span = last["elapsed_seconds"] - base["elapsed_seconds"]
        if span > 0 and key in metrics and key in base["metrics"]:
            result[label] = round((metrics[key] - base["metrics"][key]) / span, 1)
    base = next((s for s in reversed(samples) if last["elapsed_seconds"] - s["elapsed_seconds"] >= HIT_WINDOW_SECONDS),
                first)
    result["prefix_hit_recent"] = prefix_hit(base["metrics"], metrics)
    result["prefix_hit_mean"] = prefix_hit(first["metrics"], metrics)
    if ENGINE["preemptions"] in first["metrics"] and ENGINE["preemptions"] in metrics:
        result["preemptions"] = metrics[ENGINE["preemptions"]] - first["metrics"][ENGINE["preemptions"]]
    return result


def _gib(kib):
    return round(kib / 1024**2, 2)


def host_memory(proc="/proc", cgroup="/sys/fs/cgroup"):
    """Host memory now (v7, 5 October: Kaggle's ~78.6 GiB free vs PLE's pinned 47.7 GiB): used (MemTotal -
    MemAvailable), the cgroup's current / peak where readable, and the summed VmRSS of the vLLM processes (API server,
    engine core, workers; pinned PLE pages count in the engine's RSS). Monitoring only; None on any failure."""
    try:
        info = {}
        for line in Path(proc, "meminfo").read_text().splitlines():
            key, _, rest = line.partition(":")
            info[key] = int(rest.split()[0])
        out = {"used_gib": _gib(info["MemTotal"] - info["MemAvailable"]), "available_gib": _gib(info["MemAvailable"])}
        for name in ("memory.current", "memory.peak"):
            path = Path(cgroup, name)
            if path.is_file():
                out["cgroup_" + name.split(".")[1] + "_gib"] = _gib(int(path.read_text().split()[0]) // 1024)
        rss = {}
        for entry in Path(proc).iterdir():
            if not entry.name.isdigit():
                continue
            try:
                cmd = (entry / "cmdline").read_bytes().replace(b"\0", b" ").decode(errors="replace")
                if "vllm" not in cmd.lower():
                    continue
                for line in (entry / "status").read_text().splitlines():
                    if line.startswith("VmRSS:"):
                        rss[entry.name] = int(line.split()[1])
            except OSError:
                continue
        out["vllm_rss_gib"] = _gib(sum(rss.values()))
        out["vllm_processes"] = len(rss)
        return out
    except Exception:  # noqa: BLE001 - monitoring must never end the trial
        return None


class LiveProgress:
    def __init__(self, cfg, *, arc, card, monitor, games, clock=time.monotonic, log=None):
        self.cfg, self.arc, self.card, self.monitor, self.games = cfg, arc, card, monitor, games
        self.clock, self.log = clock, log
        self.score, self.score_at, self.score_error = None, None, None
        self.host_peak = {}

    def read_now(self, elapsed):
        """Read the score now (the early-stop rule's read at its time); the next progress rows carry it. Returns the
        score block read now, or None (no score, or a failed read, recorded in ``score_error``)."""
        self.score_at = self.clock()
        try:
            score = read_score(self.arc, self.card, len(self.games))
            self.score_error = None
        except Exception as exc:  # noqa: BLE001 - monitoring must never end the trial
            self.score_error = f"{type(exc).__name__}: {str(exc)[:200]}"
            return None
        if score is None:
            return None
        self.score = {**score, "at_elapsed_seconds": round(elapsed, 1)}
        return self.score

    def fields(self, elapsed):
        now = self.clock()
        if self.score_at is None or now - self.score_at >= self.cfg["score_seconds"]:
            self.read_now(elapsed)
        levels = {g.game_id: g.levels for g in self.games if g.levels}
        return {
            "score": self.score, "score_error": self.score_error,
            "levels_by_game": dict(sorted(levels.items(), key=lambda kv: (-kv[1], kv[0]))),
            "games_with_levels": len(levels),
            "completion_tokens": sum(g.clock.completion_tokens() for g in self.games),
            "gameplay_tokens": sum(g.clock.budget_tokens() for g in self.games),
            "compactions": sum(g.compactions for g in self.games),
            "compaction_seconds": round(sum(g.compaction_seconds for g in self.games), 1),
            "started_games": sum(g.clock.completion_tokens() > 0 for g in self.games),
            "actions": sum(getattr(g, "total_actions", 0) for g in self.games),
            "errors": {g.game_id: g.error for g in self.games if g.error},
            "engine": engine_snapshot(getattr(self.monitor, "samples", [])),
            "host": self.host(),
        }

    def host(self):
        now = host_memory()
        if now is None:
            return None
        for key, value in now.items():
            if key.endswith("_gib") and key != "available_gib":
                self.host_peak[key] = max(value, self.host_peak.get(key, 0))
        return {**now, "peak": dict(self.host_peak)}


def summary_line(progress):
    """One compact dict for a printed line (the bootstrap and the watcher)."""
    live = progress.get("live") or {}
    score = live.get("score") or {}
    engine = live.get("engine") or {}
    return {"elapsed_seconds": progress.get("elapsed_seconds"), "phase": progress.get("phase"),
            "score": None if score.get("score") is None else round(score["score"], 4),
            "score_at": score.get("at_elapsed_seconds"), "levels": progress.get("levels"),
            "games_with_levels": live.get("games_with_levels"), "active": len(progress.get("active") or []),
            "completion_tokens": live.get("completion_tokens"),
            "tok_s_recent": engine.get("tokens_per_second_recent"), "tok_s_mean": engine.get("tokens_per_second_mean"),
            "running": engine.get("running"), "waiting": engine.get("waiting"),
            "kv": engine.get("kv_cache_fraction"), "prefix_hit": engine.get("prefix_hit_recent"),
            "prefix_hit_mean": engine.get("prefix_hit_mean"),
            "preemptions": engine.get("preemptions"),
            "host_peak": (live.get("host") or {}).get("peak"),
            "loop_rule": progress.get("loop_rule"),
            "top_levels": dict(list((live.get("levels_by_game") or {}).items())[:6])}


def last_lines(path, lines=300, max_bytes=512 * 1024):
    """The last ``lines`` lines of a (possibly large, growing) log file, reading at most ``max_bytes``."""
    path = Path(path)
    if not path.is_file():
        return None
    with path.open("rb") as handle:
        handle.seek(0, 2)
        size = handle.tell()
        handle.seek(max(0, size - max_bytes))
        data = handle.read()
    return b"\n".join(data.splitlines()[-lines:]).decode(errors="replace") + "\n"


def publish(source, dest_dir, trial, *, log=print, server_log=None):
    """Copy a trial's ``progress.json`` to ``dest_dir/live`` and print one line (the bootstrap's tailer); with
    ``server_log``, also its last lines (a stall dump or an engine error shows mid-run)."""
    live = Path(dest_dir) / "live"
    if server_log is not None:
        text = last_lines(server_log)
        if text is not None:
            live.mkdir(parents=True, exist_ok=True)
            (live / "server-tail.log").write_text(text)
    path = Path(source)
    if not path.is_file():
        return None
    try:
        progress = json.loads(path.read_text())
    except (OSError, ValueError):
        return None  # mid-write; the next tick reads it
    live.mkdir(parents=True, exist_ok=True)
    record = {"trial": trial, "epoch": round(time.time(), 1), **progress}
    (live / "progress.json").write_text(json.dumps(record, indent=1))
    with (live / "progress.jsonl").open("a") as file:
        file.write(json.dumps(record) + "\n")
    log(json.dumps({"event": "live_progress", "trial": trial, **summary_line(progress)}), flush=True)
    return record


def tail(source, dest_dir, trial, stop, interval, *, log=print, server_log=None):
    """Bootstrap thread: ``publish`` every ``interval`` s until ``stop`` is set, then once more."""
    while not stop.wait(interval):
        try:
            publish(source, dest_dir, trial, log=log, server_log=server_log)
        except Exception as exc:  # noqa: BLE001 - never end the run over monitoring
            log(json.dumps({"event": "live_progress_error", "error": f"{type(exc).__name__}: {exc}"[:300]}),
                flush=True)
    try:
        publish(source, dest_dir, trial, log=log, server_log=server_log)
    except Exception:  # noqa: BLE001
        pass
