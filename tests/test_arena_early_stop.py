"""Graceful early stop of an arena trial (arena.early_stop): the 1 h score rule and the manual stop request, with fake
scorecards, a fake clock and fake agents on the real trial loop; the build flag and the Modal stop watcher."""

import asyncio
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from arena import early_stop as es
from arena import trial as trial_module
from arena.clock import GlobalClock
from arena.trial import Trial

ROOT = Path(__file__).resolve().parents[1]
STUDY = json.loads((ROOT / "configs/arena.json").read_text())


def test_settings_are_validated():
    assert es.settings(None) is None and es.settings(False) is None
    cfg = es.settings({"at_seconds": 3600, "score_above": 1.68}, wall_seconds=7200)
    assert cfg == {"at_seconds": 3600, "score_above": 1.68, "retry_seconds": 300}
    for bad in ({"at_seconds": 3600}, {"at_seconds": 3600, "score_above": -1}, {"at_seconds": 0, "score_above": 1},
                {"at_seconds": 3600, "score_above": 1, "extra": 1}, {"at_seconds": 3600.5, "score_above": 1}, "1h"):
        with pytest.raises(ValueError):
            es.settings(bad)
    with pytest.raises(ValueError, match="before the wall"):
        es.settings({"at_seconds": 7200, "score_above": 1.68}, wall_seconds=7200)


def rule(score, tmp_path, **cfg):
    reads = []

    def read(elapsed):
        reads.append(elapsed)
        value = score(elapsed) if callable(score) else score
        return None if value is None else {"score": value, "at_elapsed_seconds": elapsed}

    events = []
    stop = es.EarlyStop(es.settings({"at_seconds": 3600, "score_above": 1.68, **cfg}), read_score=read,
                        paths=es.stop_paths(tmp_path, environ={}), log=events.append)
    return stop, reads, events


def test_the_first_read_at_or_after_one_hour_decides(tmp_path):
    stop, reads, events = rule(2.0, tmp_path)
    assert stop.check(3599.9) is None and reads == []          # no read before the hour
    assert stop.check(3600.4) == "early_stop" and reads == [3600.4]
    assert stop.check(3601.4) == "early_stop" and reads == [3600.4]   # decided once
    summary = stop.summary()
    assert summary["decision"] == "stop" and summary["reason"] == "early_stop"
    assert summary["score"]["score"] == 2.0 and summary["decided_at_elapsed_seconds"] == 3600.4
    assert events[-1]["decision"] == "stop" and events[-1]["score_above"] == 1.68


def test_a_score_at_or_below_the_threshold_continues_to_the_wall(tmp_path):
    for score in (1.68, 1.2):
        stop, reads, _ = rule(score, tmp_path)
        assert stop.check(3600.0) is None and stop.decision == "continue"
        assert stop.check(5000.0) is None and len(reads) == 1   # never read again


def test_a_failed_read_is_retried_then_the_trial_continues(tmp_path):
    stop, reads, events = rule(lambda t: None if t < 3700 else 5.0, tmp_path, retry_seconds=60)
    assert stop.check(3600.0) is None and stop.decision is None
    assert stop.check(3630.0) is None and stop.decision is None
    assert stop.check(3661.0) is None and stop.decision == "continue"
    assert stop.check(3700.0) is None and len(reads) == 3           # decided: no late stop on a later good score
    assert events[-1]["reason"] == "no score"

    def broken(_):
        raise RuntimeError("gateway down")

    stop = es.EarlyStop(es.settings({"at_seconds": 10, "score_above": 1.0, "retry_seconds": 0}), read_score=broken,
                        paths=[])
    assert stop.check(11.0) is None and stop.decision == "continue" and "gateway down" in stop.error


def test_a_stop_request_ends_the_trial_whatever_the_score(tmp_path):
    local = tmp_path / "volume-mirror"
    stop = es.EarlyStop(None, read_score=lambda e: {"score": 0.0}, paths=es.stop_paths(
        tmp_path, environ={es.STOP_ENV: str(local)}))
    assert [str(p) for p in stop.paths] == [str(tmp_path / "stop-request"), str(local)]
    assert stop.check(10.0) is None
    es.request_stop(local, "enough evidence")
    assert stop.check(12.0) == "stop_request"
    assert stop.summary()["stop_request"]["path"] == str(local)
    assert "coordinator" in stop.summary()["stop_request"]["text"]


def test_end_trial_is_a_normal_wall_end():
    now = [100.0]
    wall = GlobalClock(7200, now=lambda: now[0])
    wall.release()
    now[0] += 3601.0
    assert es.end_trial(wall) == pytest.approx(3601.0) and wall.remaining() == 0.0
    assert wall.feedback()["remaining_seconds"] == 0.0


# --- the real trial loop with fake agents -----------------------------------------------------------------------------

class ScoreArcade:
    """A research-gateway arcade: the open card's score is ``score``; ``peek_score`` is a class attribute."""

    score = 0.0

    def __init__(self, games):
        self.games, self.peeks = games, 0

    def get_environments(self):
        return [SimpleNamespace(game_id=g) for g in self.games]

    def open_scorecard(self, tags):
        return "card"

    def peek_score(self, card):
        self.peeks += 1
        envs = [SimpleNamespace(id=g, score=self.score) for g in self.games]
        return SimpleNamespace(environments=envs, total_levels_completed=3)

    def close_scorecard(self, card):
        envs = [SimpleNamespace(id=g, score=self.score) for g in self.games]
        return SimpleNamespace(environments=envs, model_dump=lambda **kw: {"environments": [e.id for e in envs]})


class OfflineMonitor:
    def __init__(self, output, base_url):
        self.start, self.samples = None, []

    async def sample(self):
        return None

    async def run(self):
        await asyncio.Event().wait()

    def summary(self):
        return {"valid": True, "preemptions": 0, "accepted_generation_tokens": 10}


class Worker:
    def __init__(self, game):
        self.game = game

    async def run(self):
        self.game.ready.set()
        await asyncio.Event().wait()

    def result(self):
        return {"game_id": self.game.game_id, "status": "time_limit", "error": None, "levels": 0, "total_levels": 6,
                "actions": 1, "model_calls": 1, "prompt_tokens": 10, "completion_tokens": 5}


def fake_trial(tmp_path, monkeypatch, *, score, wall=3.0, early=None, live=False):
    games = STUDY["games"][:3]
    study = {**STUDY, "games": games, "wall_seconds": wall}
    if early is not None:
        study["early_stop"] = early
    if live:
        study["live_progress"] = {"interval_seconds": 60, "score_seconds": 300}
    arcade = ScoreArcade(games)
    arcade.score = score
    monkeypatch.setattr(trial_module, "kv_pool_tokens", lambda server_dir: 291_900)
    monkeypatch.setattr(trial_module, "GpuMonitor", OfflineMonitor)

    async def agents(self, records, gate, window, wall_clock, card, arc):
        workers = [Worker(r) for r in records]
        self.config_dict = {"reasoning_effort": "xhigh", "temperature": 0.7, "gameplay_seconds": wall}

        async def close():
            return None

        return workers, (lambda: [w.result() for w in workers]), close

    monkeypatch.setattr(Trial, "duck_agents", agents)
    trial = Trial(study, study["trials"][0], source=".", games_dir="games", output=tmp_path / "t", server_dir=tmp_path,
                  base_url="http://127.0.0.1:1234/v1", log=lambda *a, **k: None, arcade=arcade)
    return trial, arcade


async def test_the_trial_stops_at_the_rule_when_the_score_beats_the_threshold(tmp_path, monkeypatch):
    trial, arcade = fake_trial(tmp_path, monkeypatch, score=2.5, early={"at_seconds": 1, "score_above": 1.68},
                               live=True)
    report = await trial.run()
    assert report["end_reason"] == "early_stop" and report["early_stop"]["decision"] == "stop"
    assert report["early_stop"]["score"]["score"] == pytest.approx(2.5)
    assert 1.0 <= report["wall_seconds"] < 2.5 and report["wall"]["configured_seconds"] == 3.0
    assert report["wall"]["ended_seconds"] == pytest.approx(report["wall_seconds"], abs=0.2)
    # Everything a wall end writes is there: the report, the scorecard, the progress row with the decision.
    out = tmp_path / "t"
    assert (out / "arena.json").is_file() and (out / "scorecard.json").is_file()
    progress = json.loads((out / "progress.json").read_text())
    assert progress["early_stop"]["decision"] == "stop" and progress["early_stop"]["reason"] == "early_stop"
    assert progress["live"]["score"]["score"] == pytest.approx(2.5)   # the rule's read is the live score
    assert '"early_stop"' in (out / "admission.jsonl").read_text()


async def test_the_trial_runs_to_its_wall_below_the_threshold(tmp_path, monkeypatch):
    trial, _ = fake_trial(tmp_path, monkeypatch, score=1.0, early={"at_seconds": 1, "score_above": 1.68})
    report = await trial.run()
    assert report["end_reason"] == "wall" and report["early_stop"]["decision"] == "continue"
    assert report["wall_seconds"] >= 3.0


async def test_a_stop_request_ends_the_trial_gracefully(tmp_path, monkeypatch):
    trial, _ = fake_trial(tmp_path, monkeypatch, score=0.0, wall=30.0)
    monkeypatch.setenv(es.STOP_ENV, str(tmp_path / "mirror"))

    async def request():
        await asyncio.sleep(0.8)
        es.request_stop(tmp_path / "mirror", "coordinator")

    task = asyncio.create_task(request())
    report = await trial.run()
    await task
    assert report["end_reason"] == "stop_request" and report["wall_seconds"] < 5.0
    assert report["early_stop"]["stop_request"]["path"] == str(tmp_path / "mirror")
    assert (tmp_path / "t/scorecard.json").is_file()


async def test_no_early_stop_block_without_the_setting_or_a_request(tmp_path, monkeypatch):
    trial, _ = fake_trial(tmp_path, monkeypatch, score=5.0, wall=1.2)
    monkeypatch.delenv(es.STOP_ENV, raising=False)
    report = await trial.run()
    assert report["end_reason"] == "wall" and "early_stop" not in report


# --- build and Modal ---------------------------------------------------------------------------------------------------

def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_the_modal_wrapper_mirrors_the_volume_stop_request():
    source = (ROOT / "scripts/modal_validation.py").read_text()
    assert 'os.environ["ARENA_STOP_REQUEST"] = STOP_LOCAL' in source
    assert "runs.listdir(folder)" in source and 'f"validation/{run_id}/arena/live"' in source
    modal = load("modal_validation_early_stop", ROOT / "scripts/modal_validation.py")
    assert modal.STOP_FILE == es.STOP_FILE and modal.STOP_LOCAL.startswith("/tmp/")
    for test in ("hero-v2-128k-int4", "hero-v2-streams"):
        assert modal.validation_timeout("arena", test) == 9600
        assert modal.arena_arguments(test) == ["--trial", "duck-transfer"]
    rate = 0.000842 + 8 * 0.0000131 + 192 * 0.00000222
    assert round((9600 + 60) * rate * 1.15, 2) == 15.25
