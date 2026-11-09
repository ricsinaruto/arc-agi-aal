"""The arena trial driver and CLI with fake agents, arcade, pool and monitor."""

import asyncio
import json
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import pytest

from arena import run as run_module
from arena import trial as trial_module
from arena.report import render
from arena.trial import Trial

STUDY = json.loads((Path(__file__).resolve().parents[1] / "configs/arena.json").read_text())


def test_gate_events_have_wall_and_epoch_timestamps():
    import io

    from arena.clock import GlobalClock

    trial = object.__new__(Trial)
    trial.admission_log = io.StringIO()
    now = [100.0]
    trial.wall = GlobalClock(1800, now=lambda: now[0])
    trial.wall.release()
    now[0] += 4.5
    trial.gate_log({'event': 'release', 'game': 'a'})
    trial.gate_log({'event': 'park', 'wall_seconds': 4.25})
    rows = [json.loads(line) for line in trial.admission_log.getvalue().splitlines()]
    assert [r['wall_seconds'] for r in rows] == [4.5, 4.25]
    assert all(r['epoch'] > 0 for r in rows)


class FakeArcade:
    def __init__(self, games):
        self.games = games
        self.closed = None

    def get_environments(self):
        return [SimpleNamespace(game_id=g) for g in self.games]

    def open_scorecard(self, tags):
        return "card"

    def close_scorecard(self, card):
        envs = [SimpleNamespace(id=g, score=float(10 * i)) for i, g in enumerate(self.games)]
        self.closed = SimpleNamespace(environments=envs, model_dump=lambda **kw: {"environments": [e.id for e in envs]})
        return self.closed


class OfflineMonitor:
    def __init__(self, output, base_url):
        self.start = None

    async def sample(self):
        return None

    async def run(self):
        await asyncio.Event().wait()

    def summary(self):
        return {"valid": True, "preemptions": 0, "accepted_generation_tokens": 10}


class FakeWorker:
    """Plays until cancelled; reports a level after ``clears_at`` active seconds."""

    def __init__(self, game, clears_at=None):
        self.game, self.clears_at = game, clears_at
        self.closed = False

    async def run(self):
        self.game.ready.set()
        while True:
            await asyncio.sleep(0.02)
            if self.clears_at is not None and self.game.levels == 0 and self.game.clock.active_seconds() >= self.clears_at:
                self.game.report_level(1, self.game.clock.global_clock.elapsed())

    def result(self):
        return {"game_id": self.game.game_id, "status": "time_limit", "error": None, "levels": self.game.levels,
                "total_levels": 6, "actions": 3, "model_calls": 2, "prompt_tokens": 100, "completion_tokens": 50}

    async def close(self):
        self.closed = True


async def test_trial_runs_the_scheduler_over_fake_agents(tmp_path, monkeypatch):
    games = STUDY["games"][:5]
    study = {**STUDY, "games": games, "wall_seconds": 1.2, "smoke": {**STUDY["smoke"], "games": 3, "wall_seconds": 0.6}}
    monkeypatch.setattr(trial_module, "make_arcade", lambda mode, games_dir, rec: FakeArcade(games))
    monkeypatch.setattr(trial_module, "kv_pool_tokens", lambda server_dir: 291_900)
    monkeypatch.setattr(trial_module, "GpuMonitor", OfflineMonitor)
    workers = []

    async def fake_duck_agents(self, records, gate, window, wall, card, arc):
        assert window == 131072 and gate.capacity == 262_710
        for record in records:
            workers.append(FakeWorker(record, clears_at=0.1 if record.index in (1, 3) else None))
        self.config_dict = {"reasoning_effort": "xhigh", "temperature": 0.7}

        async def close():
            for w in workers:
                await w.close()

        return workers, (lambda: [w.result() for w in workers]), close

    monkeypatch.setattr(Trial, "duck_agents", fake_duck_agents)
    trial = Trial(study, study["trials"][0], source=".", games_dir="games", output=tmp_path / "t",
                  server_dir=tmp_path, base_url="http://127.0.0.1:1234/v1", log=lambda *a, **k: None)
    report = await trial.run()
    assert report["status"] == "complete" and report["preemption_free"] and report["sessions"] == 5
    assert report["score_denominator_games"] == 5
    assert report["window_choice"]["window"] == 131072 and report["completion_cap"] == 16384
    assert report["scheduler"]["phase"] == "steady" and report["scheduler"]["switch_wall_seconds"] is not None
    states = {r["game_id"]: r["state"] for r in report["games"]}
    assert sum(s == "active" for s in states.values()) <= 2 and any(s == "parked" for s in states.values())
    assert report["games"][1]["time_to_level"][0]["level"] == 1 and report["games"][1]["t1_seconds"] > 0
    assert report["mean_score_equal_weight_25_games"] == 20.0  # 0,10,20,30,40 from the fake scorecard
    assert all(w.closed for w in workers)
    assert (tmp_path / "t/arena.json").is_file() and (tmp_path / "t/scorecard.json").is_file()
    assert (tmp_path / "t/admission.jsonl").read_text().count('"opening_start"') == 5
    # Smoke settings cut the game list and the wall clock.
    assert trial_module.trial_settings(study, smoke=True) == (games[:3], 0.6)


def test_cli_merges_trials_and_renders(tmp_path, monkeypatch):
    study_path = tmp_path / "arena.json"
    study_path.write_text(json.dumps(STUDY))
    calls = []

    async def fake_run_trial(study, trial, **kwargs):
        calls.append((trial["name"], kwargs["smoke"]))
        return {"status": "complete", "mean_score_equal_weight_25_games": 4.5, "levels": 3, "actions": 40,
                "scheduler": {"switch_wall_seconds": 120.0}, "engine": {"preemptions": 0}, "window_choice": {"window": 131072},
                "task_errors": []}

    async def ok(*args, **kwargs):
        return {"tokenizer_count": 5}

    monkeypatch.setattr(run_module, "run_trial", fake_run_trial)
    monkeypatch.setattr(run_module, "validate_api", ok)
    monkeypatch.setattr(run_module, "wait_idle", ok)
    monkeypatch.setattr(run_module, "hardware", lambda: {"gpu": "fake"})
    monkeypatch.setattr(run_module.signal, "setitimer", lambda *a: None)
    base = ["--source", ".", "--study", str(study_path), "--output", str(tmp_path / "out"), "--server-dir", str(tmp_path),
            "--games-dir", "games", "--notebook-start-epoch", str(time.time()), "--notebook-seconds", str(10**6)]
    run_module.main([*base, "--trial", "duck-transfer"])
    run_module.main([*base, "--trial", "duck-public", "--smoke"])
    record = json.loads((tmp_path / "out/study.json").read_text())
    assert [t["trial"]["name"] for t in record["trials"]] == ["duck-transfer", "duck-public"]
    assert calls == [("duck-transfer", False), ("duck-public", True)] and record["hardware"] == {"gpu": "fake"}
    assert record["comparison"][0]["mean_score_equal_weight_25_games"] == 4.5
    assert "| duck-transfer | complete | 4.50% | 3 | 40 | 120.0 | 0 | 131072 |" in render(record)
    assert "Score (tested games)" in render(record)
    with pytest.raises(ValueError, match="Unknown trial"):
        run_module.main([*base, "--trial", "nope"])
    assert sys.version_info >= (3, 12)


async def test_failed_worker_is_attributed_and_replaced_without_replay(tmp_path, monkeypatch):
    games = STUDY['games'][:3]
    study = {**STUDY, 'games': games, 'wall_seconds': 0.25, 'active_limit': 1,
             'opening_switch_level1_games': 1}
    monkeypatch.setattr(trial_module, 'make_arcade', lambda *args: FakeArcade(games))
    monkeypatch.setattr(trial_module, 'kv_pool_tokens', lambda *args: 291_900)
    monkeypatch.setattr(trial_module, 'GpuMonitor', OfflineMonitor)
    starts = []

    class Worker(FakeWorker):
        async def run(self):
            starts.append(self.game.game_id)
            self.game.ready.set()
            await self.game.clock.wait_active()
            if self.game.index == 0:
                self.game.report_level(1, self.game.clock.global_clock.elapsed())
                self.game.observer.tick()
                await asyncio.sleep(0.06)
                raise RuntimeError('transport failed after acknowledged action')
            await asyncio.Event().wait()

    async def agents(self, records, *args):
        workers = [Worker(g) for g in records]
        self.config_dict = {}

        async def close():
            for w in workers:
                await w.close()

        return workers, lambda: [w.result() for w in workers], close

    monkeypatch.setattr(Trial, 'duck_agents', agents)
    trial = Trial(study, study['trials'][0], source='.', games_dir='games',
                  output=tmp_path/'trial', server_dir=tmp_path, base_url='http://localhost',
                  log=lambda *a, **k: None)
    result = await trial.run()
    assert result['status'] == 'error' and result['failed_sessions'] == 1
    assert result['task_errors'] == [f'{games[0]}: RuntimeError: transport failed after acknowledged action']
    failed = result['games'][0]
    assert failed['status'] == 'error' and 'transport failed' in failed['error']
    assert failed['actions'] == 3 and failed['levels'] == 1  # Preserve completed evidence.
    assert failed['finished_reason'] == 'worker_error'
    assert starts == games  # No automatic restart/replay.
    assert any(e['event'] == 'admit' and e['reason'] == 'regrant' and e['game'] != games[0]
               for e in result['scheduler']['events'])


@pytest.mark.parametrize('status', ['action_limit', 'error', None])
async def test_returned_worker_releases_focus_slot_without_restart(tmp_path, monkeypatch, status):
    games = STUDY['games'][:3]
    study = {**STUDY, 'games': games, 'wall_seconds': 0.25, 'active_limit': 1,
             'opening_switch_level1_games': 1}
    monkeypatch.setattr(trial_module, 'make_arcade', lambda *args: FakeArcade(games))
    monkeypatch.setattr(trial_module, 'kv_pool_tokens', lambda *args: 291_900)
    monkeypatch.setattr(trial_module, 'GpuMonitor', OfflineMonitor)
    starts, returned = [], []

    class Worker(FakeWorker):
        async def run(self):
            starts.append(self.game.game_id)
            self.game.ready.set()
            await self.game.clock.wait_active()
            if self.game.index == 0:
                self.game.report_level(1, self.game.clock.global_clock.elapsed())
                self.game.observer.tick()
                await asyncio.sleep(0.05)
                returned.append(self.game.clock.active_seconds())
                # Duck catches exceptions and returns a report; not every exit raises.
                return None if status is None else self.result()
            await asyncio.Event().wait()

        def result(self):
            result = super().result()
            if self.game.index == 0 and status is not None:
                result.update(status=status, error='handled worker error' if status == 'error' else None)
            return result

    async def agents(self, games, *args):
        workers = [Worker(g) for g in games]
        self.config_dict = {}

        async def close():
            for w in workers:
                await w.close()

        return workers, lambda: [w.result() for w in workers], close

    monkeypatch.setattr(Trial, 'duck_agents', agents)
    trial = Trial(study, study['trials'][0], source='.', games_dir='games',
                  output=tmp_path/'trial', server_dir=tmp_path, base_url='http://localhost',
                  log=lambda *a, **k: None)
    result = await trial.run()
    ended = result['games'][0]
    assert ended['state'] == 'finished' and ended['finished_reason'] == 'worker_returned'
    assert ended['active_seconds'] < 0.15 and ended['active_seconds'] >= returned[0] - 0.002
    assert ended['actions'] == 3 and ended['levels'] == 1
    assert result['status'] == ('error' if status == 'error' else 'complete')
    assert starts == games  # Terminated code and actions are never replayed.
    events = result['scheduler']['events']
    assert any(e['event'] == 'finish' and e['game'] == games[0] for e in events)
    assert any(e['event'] == 'admit' and e['reason'] == 'regrant' and e['game'] != games[0]
               for e in events)


@pytest.mark.parametrize('kind', ['duck-transfer'])
@pytest.mark.parametrize('bad_report', [False, True])
@pytest.mark.parametrize('block_admission', [False, True])
async def test_archive_metrics_snapshot_after_drain_before_close(tmp_path, monkeypatch, kind, bad_report, block_admission):
    import sqlite3

    games = STUDY['games'][:1]
    study = {**STUDY, 'games': games, 'wall_seconds': 0.06}
    if block_admission:
        study['cache_admission'] = 'hybrid_blocks'
        budget = SimpleNamespace(capacity=199, cost=lambda tokens: 21 + tokens // 1600,
                                 layout={'rule': 'test measured blocks'})
        monkeypatch.setattr(trial_module.BlockBudget, 'load', lambda *args: budget)
    arcade = FakeArcade(games)
    monkeypatch.setattr(trial_module, 'make_arcade', lambda *a: arcade)
    monkeypatch.setattr(trial_module, 'kv_pool_tokens', lambda *a: 291_900)
    monkeypatch.setattr(trial_module, 'GpuMonitor', OfflineMonitor)
    db = sqlite3.connect(tmp_path / 'evidence.sqlite')
    db.execute('CREATE TABLE evidence (value INTEGER)')
    workers = []

    class Worker(FakeWorker):
        async def run(self):
            self.game.ready.set()
            try:
                await asyncio.Event().wait()
            finally:
                db.execute('INSERT INTO evidence VALUES (17)')
                db.commit()

    async def agents(self, records, *args):
        if block_admission:
            assert args[0].budget is budget and args[0].capacity == 199
        worker = Worker(records[0])
        workers.append(worker)
        self.config_dict = {}

        def rows():
            # Must include evidence finalized by cancellation and still be readable.
            actions = db.execute('SELECT SUM(value) FROM evidence').fetchone()[0]
            assert actions == 17
            if bad_report:
                raise ValueError('report failure')
            return [{**worker.result(), 'actions': actions}]

        async def close():
            await worker.close()
            db.close()

        return workers, rows, close

    monkeypatch.setattr(Trial, 'duck_agents', agents)
    trial = Trial(study, {'name': kind, 'kind': kind}, source='.', games_dir='games',
                  output=tmp_path / 'trial', server_dir=tmp_path, base_url='http://localhost',
                  log=lambda *a, **k: None)
    if bad_report:
        with pytest.raises(ValueError, match='report failure'):
            await trial.run()
    else:
        result = await trial.run()
        assert result['actions'] == 17 and result['status'] == 'complete'
    assert workers[0].closed and arcade.closed is not None
    with pytest.raises(sqlite3.ProgrammingError, match='closed database'):
        db.execute('SELECT 1')
