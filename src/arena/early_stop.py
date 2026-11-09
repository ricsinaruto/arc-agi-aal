"""Graceful early stop of an arena trial ("we will run these as 2-hour, but stop after 1 hour if
score is higher than hero-v2 @ 1 hour"; docs/validation-20260927/hero-v2-next-runs.md).

``study["early_stop"] = {"at_seconds": 3600, "score_above": 1.68}``: at the first tick at or after ``at_seconds`` of
gameplay the trial reads the open scorecard (the same read as live progress: the arena's percent score, the mean over
the roster). Above ``score_above`` it ends the trial as a normal wall end: the wall clock's length becomes the elapsed
time, so the agents see no time left, the scheduler's wall end, the scorecard close, the report and every evidence
file are written as at the configured wall. Otherwise the trial runs to its wall. A failed read is retried each tick
for ``retry_seconds``; then the trial continues (decision "continue", reason "no score").

A manual stop (the operator): a file named ``stop-request`` in the trial folder, or at the path in the
``ARENA_STOP_REQUEST`` environment variable (the Modal wrapper creates it when ``live/stop-request`` appears on the runs
volume), ends the trial the same graceful way at the next tick, whatever the score.

The rule reads only the trial's own scorecard (the local research gateway or the offline arcade; never the hidden
gateway) and changes nothing an agent or the scheduler sees before the stop.
"""

import json
import math
import os
import time
from pathlib import Path

DEFAULTS = {"at_seconds": 3600, "score_above": None, "retry_seconds": 300, "always": False}
STOP_FILE = "stop-request"
STOP_ENV = "ARENA_STOP_REQUEST"


def settings(spec, wall_seconds=None):
    """Validated ``early_stop`` settings, or None."""
    if spec in (None, False):
        return None
    if not isinstance(spec, dict):
        raise ValueError("early_stop is {at_seconds, score_above[, retry_seconds]}")
    cfg = {**DEFAULTS, **spec}
    if set(cfg) != set(DEFAULTS):
        raise ValueError(f"Unknown early_stop settings: {sorted(set(cfg) - set(DEFAULTS))}")
    if type(cfg["at_seconds"]) is not int or cfg["at_seconds"] < 1:
        raise ValueError("early_stop.at_seconds is a positive integer (seconds of gameplay)")
    if wall_seconds is not None and cfg["at_seconds"] >= wall_seconds:
        raise ValueError("early_stop.at_seconds must be before the wall")
    if type(cfg["always"]) is not bool:
        raise ValueError("early_stop.always is true or false")
    score = cfg["score_above"]
    if cfg["always"]:
        # An unconditional run stop (hero-v2-blocks, 29 September): the trial ends at at_seconds whatever the score, and
        # the wall the scheduler and the agents see stays the study's (e.g. 30,600 s of a 7,200 s research run).
        if score is not None:
            raise ValueError("early_stop.always stops whatever the score: no score_above")
    elif type(score) not in (int, float) or not math.isfinite(score) or not 0 <= score < 100:
        raise ValueError("early_stop.score_above is the arena's percent score (0..100)")
    if type(cfg["retry_seconds"]) is not int or not 0 <= cfg["retry_seconds"] <= 1800:
        raise ValueError("early_stop.retry_seconds is 0..1800")
    if not cfg["always"]:
        cfg.pop("always")  # the score rule's recorded settings are unchanged
    return cfg


def stop_paths(output, environ=os.environ):
    paths = [Path(output) / STOP_FILE]
    if environ.get(STOP_ENV):
        paths.append(Path(environ[STOP_ENV]))
    return paths


class EarlyStop:
    """Checked every trial tick: ``check(elapsed)`` returns the end reason ("early_stop", "stop_request") or None.

    ``read_score(elapsed)`` returns the live score block ({"score": ..., ...}) or None; ``paths`` are the stop-request
    files. Without ``cfg`` only the stop request is watched."""

    def __init__(self, cfg, *, read_score, paths, log=None, now=time.time):
        self.cfg, self.read_score, self.paths, self.log, self.now = cfg, read_score, list(paths), log, now
        self.decision = None       # "stop" | "continue" once the rule has been evaluated
        self.reason = None         # the end reason when the trial stops early
        self.score = None          # the score read the decision used
        self.decided_at = None
        self.first_attempt = None
        self.attempts = 0
        self.error = None
        self.stop_request = None

    def _event(self, entry):
        if self.log is not None:
            self.log({"event": "early_stop", **entry})

    def _requested(self):
        for path in self.paths:
            try:
                if path.exists():
                    try:
                        text = path.read_text()[:500]
                    except OSError:
                        text = ""
                    return {"path": str(path), "text": text.strip()}
            except OSError:
                continue
        return None

    def check(self, elapsed):
        if self.reason is not None:
            return self.reason
        request = self._requested()
        if request is not None:
            self.stop_request = {**request, "at_elapsed_seconds": round(elapsed, 1), "epoch": round(self.now(), 1)}
            self.reason = "stop_request"
            self._event({"reason": self.reason, **self.stop_request})
            return self.reason
        cfg = self.cfg
        if cfg is None or self.decision is not None or elapsed < cfg["at_seconds"]:
            return None
        if self.first_attempt is None:
            self.first_attempt = elapsed
        self.attempts += 1
        if cfg.get("always"):
            # Unconditional stop: one score read for the record (the live-progress read, so progress shows the score
            # at the stop), then a normal wall end whatever it says.
            try:
                score = self.read_score(elapsed)
            except Exception as exc:  # noqa: BLE001 - the stop never depends on the read
                score, self.error = None, f"{type(exc).__name__}: {str(exc)[:200]}"
            self.score, self.decided_at = (None if score is None else dict(score)), round(elapsed, 1)
            self.decision, self.reason = "stop", "stop_at"
            self._event({"decision": self.decision, "reason": self.reason, "always": True,
                         "score": None if score is None else score.get("score"), "at_elapsed_seconds": self.decided_at})
            return self.reason
        try:
            score = self.read_score(elapsed)
            self.error = None
        except Exception as exc:  # noqa: BLE001 - a failed read never ends the trial
            score, self.error = None, f"{type(exc).__name__}: {str(exc)[:200]}"
        if score is None or score.get("score") is None:
            if elapsed - self.first_attempt >= cfg["retry_seconds"]:
                self.decision, self.decided_at = "continue", round(elapsed, 1)
                self._event({"decision": self.decision, "reason": "no score", "attempts": self.attempts,
                             "error": self.error, "at_elapsed_seconds": self.decided_at})
            return None
        self.score, self.decided_at = dict(score), round(elapsed, 1)
        above = float(score["score"]) > float(cfg["score_above"])
        self.decision = "stop" if above else "continue"
        self._event({"decision": self.decision, "score": score["score"], "score_above": cfg["score_above"],
                     "at_elapsed_seconds": self.decided_at, "attempts": self.attempts})
        if above:
            self.reason = "early_stop"
        return self.reason

    def summary(self):
        return {"settings": None if self.cfg is None else dict(self.cfg), "decision": self.decision,
                "reason": self.reason, "score": self.score, "decided_at_elapsed_seconds": self.decided_at,
                "attempts": self.attempts, "error": self.error, "stop_request": self.stop_request,
                "stop_paths": [str(p) for p in self.paths]}


def end_trial(wall, elapsed=None):
    """End the trial's wall clock now (a normal wall end): its length becomes the elapsed gameplay time."""
    wall.seconds = min(wall.seconds, wall.elapsed() if elapsed is None else elapsed)
    return wall.seconds


def request_stop(path, reason):
    """Write a stop request (tests and local runs; on Modal ``scripts/arena/request_stop.py`` writes the volume)."""
    Path(path).write_text(json.dumps({"reason": reason, "epoch": round(time.time(), 1)}) + "\n")
