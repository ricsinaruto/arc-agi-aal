"""Shared watcher backtests on 24 Oct logs: hero-v6 (starved play), c7b (host-RAM death), the PSFG boot crash; healthy
counterparts of the same kind stay quiet (ovf3, c6, a healthy bootstrap log)."""
import importlib.util
import json
from pathlib import Path

import pytest

from arc_infra.runops.watch import Snapshot, Watch, validate_plan

FIX = Path(__file__).parent / "fixtures" / "watch"
ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("ops_watch", ROOT / "tools/ops/watch.py")
ops_watch = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ops_watch)


def rows(name):
    return [json.loads(x) for x in (FIX / name).read_text().splitlines() if x.strip()]


def t0(r):
    return min(x["t"] for x in r)


def rules(alerts):
    return sorted({a.rule for _, a in alerts})


RL_HERO = {"targets": {"infra.running": {"plan": 38, "steady": True}, "infra.gen_tok_s": {"plan": 900, "steady": True},
                       "infra.gpu_util.*": 85}, "boot_s": 600, "cadence_s": 60, "util_exempt": ["infra.gpu_util.1"]}
PTH_OPD = {"targets": {"infra.gpu_util.*": 80}, "boot_s": 600, "cadence_s": 60, "host": {"next_alloc_gib": 443}}


def test_hero_v6_starved_play_alerts_at_10_min_of_play():
    r = rows("hero-v6.jsonl")
    out = ops_watch.backtest({"run": "hero-v6", "launched": t0(r), **RL_HERO}, r)
    first = [(m, a) for m, a in out if a.rule == "plan@10m"]
    assert first and first[0][0] <= 21  # play from 10 min (boot), checked at 20 min after launch
    texts = " ".join(a.text for _, a in first)
    assert "infra.running" in texts and "infra.gen_tok_s" in texts
    assert "rate:infra.running" in rules(out)


def test_ovf3_healthy_rl_is_quiet():
    r = rows("ovf3-80m.jsonl")
    plan = {"run": "ovf3", "launched": t0(r), "boot_s": 1260, "cadence_s": 60,
            "targets": {"infra.running": 40, "infra.gen_tok_s": 1000, "infra.gpu_util.*": 85, "infra.vram_gib.*": 250}}
    out = ops_watch.backtest(plan, r)
    assert [a for _, a in out if a.rule != "run-alert"] == []


C7B_LAUNCH = 1792827452.0  # 07:37:32, first attempt (W1024 07:37:31 in its log)
C7B_RESTART = 1792828518.0  # 07:55:18: Modal restarted it; the attempt died during the first checkpoint copy


def test_c7b_host_ram_before_death_then_silence():
    r = rows("opdS-c7b-attempt1.jsonl")
    out = ops_watch.backtest({"run": "c7b", "launched": C7B_LAUNCH, **PTH_OPD, "targets": {}}, r,
                             beat_until=C7B_RESTART - 60, until=C7B_RESTART + 1800)  # the heartbeat dies with it
    host = [m for m, a in out if a.rule == "host-ram"]
    silent = [m for m, a in out if a.rule == "silent"]
    death = (C7B_RESTART - C7B_LAUNCH) / 60
    assert host and host[0] < death  # warned before the death
    assert silent and silent[0] <= death + 11
    assert "144.2 GiB < 1.2 x next allocation 443" in next(a.text for _, a in out if a.rule == "host-ram")


def test_c6_healthy_pth_is_quiet():
    r = rows("opdS-c6.jsonl")
    out = ops_watch.backtest({"run": "c6", "launched": t0(r), **PTH_OPD}, r, beat_until=None, live_until=None)
    assert out == []


def test_c7b_rows_only_checkpoint_would_not_alert():
    r = rows("opdS-c7b-attempt1.jsonl")  # 184 GiB rows-only checkpoint over 3 ranks, written one rank at a time
    plan = {"run": "c7b", "launched": C7B_LAUNCH, **PTH_OPD, "targets": {}, "host": {"next_alloc_gib": 62}}
    assert "host-ram" not in rules(ops_watch.backtest(plan, r))


def test_psfg_boot_crash_and_healthy_bootstrap():
    bad = (FIX / "psfg-bootstrap.log").read_text()
    good = (FIX / "healthy-bootstrap.log").read_text()
    w = Watch({"run": "psfg", "launched": 0, "boot_s": 1500})
    crash = [a for a in w.check(Snapshot(now=60, rows=[], live=True, log_text=bad)) if a.rule == "crash"]
    assert any("SyntaxError" in a.text for a in crash)
    assert Watch({"run": "ok", "launched": 0}).check(Snapshot(now=60, rows=[], live=True, log_text=good)) == []


def test_boot_overdue_and_gone():
    w = Watch({"run": "x", "launched": 0, "boot_s": 600, "targets": {"infra.running": 10}})
    assert w.check(Snapshot(now=800, rows=[], live=True)) == []
    assert [a.rule for a in w.check(Snapshot(now=901, rows=[], live=True))] == ["boot"]
    assert [a.rule for a in w.check(Snapshot(now=960, rows=[], live=False))] == ["unclean-end"]
    assert w.check(Snapshot(now=990, rows=[], live=False, ended=True)) == []
    w2 = Watch({"run": "x", "launched": 0, "boot_s": 600})
    assert [a.rule for a in w2.check(Snapshot(now=60, rows=[], live=False, clean_end=True))] == ["gone"]  # settle it
    assert w2.check(Snapshot(now=120, rows=[], live=False, clean_end=True, ended=True)) == []


def test_c7d_app_died_at_19s_is_unclean_not_end():
    """24 Oct: c7d (ap-Xx01u8RJBBKpkElX9HxsaN) died 19 s in on a ValueError; the watcher printed END and exited 0."""
    log = (FIX / "opdS-c7d-app.log").read_text()
    plan = {"run": "c7d", "launched": 0, "boot_s": 600, "platform": "modal", "app_id": "ap-Xx01u8RJBBKpkElX9HxsaN",
            "targets": {"infra.gpu_util.*": 80}}
    out = ops_watch.backtest(plan, [], log_text=log, live_until=19, until=600)
    assert "unclean-end" in rules(out) and "crash" in rules(out)
    assert min(m for m, a in out if a.rule == "unclean-end") <= 1.0
    clean = ops_watch.backtest(plan, [], log_text="step 400 saved\nRUNOPS END ok\n", live_until=19, until=600)
    assert "unclean-end" not in rules(clean)


def test_silence_needs_two_signals_and_feed_lag():
    r = [{"t": 0, "m": {"infra.running": 5}}]
    w = Watch({"run": "x", "launched": 0, "boot_s": 0, "cadence_s": 60})
    assert w.check(Snapshot(now=700, rows=r, live=True)) == []  # one signal only
    assert [a.rule for a in w.check(Snapshot(now=700, rows=r, live=True, remote_age=10))] == ["feed"]
    assert [a.rule for a in w.check(Snapshot(now=760, rows=r, live=True, remote_age=700))] == ["silent"]


def test_restart_incident_relayed_once_and_reminders():
    w = Watch({"run": "x", "launched": 0, "boot_s": 600})
    inc = [{"id": "x/a1", "cause": "host-memory", "blocked": True, "why": "no checkpoint"}]
    assert [a.rule for a in w.check(Snapshot(now=10, rows=[], live=True, incidents=inc))] == ["restart"]
    assert w.check(Snapshot(now=20, rows=[], live=True, incidents=inc)) == []
    w2 = Watch({"run": "y", "launched": 0, "boot_s": 60})
    assert len(w2.check(Snapshot(now=100, rows=[], live=True))) == 1
    assert w2.check(Snapshot(now=200, rows=[], live=True)) == []  # still active: no repeat
    assert len(w2.check(Snapshot(now=1901, rows=[], live=True))) == 1  # 30-min reminder


def test_validate_plan():
    ok = {"run": "x", "launched": 1, "boot_s": 600, "targets": {"infra.running": {"plan": 38, "steady": True}}}
    assert validate_plan({"plan": ok}) == ok
    with pytest.raises(ValueError):
        validate_plan({"run": "x", "launched": 1})
    with pytest.raises(ValueError):
        validate_plan({**ok, "tragets": {}})
    with pytest.raises(ValueError):
        validate_plan({**ok, "platform": "modal"})


def test_arena_throughput_healthy_is_quiet():
    """c100 300K throughput run (psfgab v2, healthy): arena progress rows through the arena adapter."""
    from arc_infra.runops import arena

    prog = rows("psfgab-v2-progress.jsonl")
    r = arena.progress_rows(prog)
    plan = {"run": "psfgab-v2", "launched": r[0]["t"] - 60, "boot_s": 300, "cadence_s": 60,
            "targets": {"infra.running": {"plan": 40, "steady": True}, "infra.gen_tok_s": {"plan": 1500, "steady": True}}}
    assert ops_watch.backtest(plan, r, beat_until=None) == []
    starved = [{**x, "m": {**x["m"], "infra.running": 4.0}} for x in r]  # the same run at 4 live streams
    assert "plan@10m" in rules(ops_watch.backtest(plan, starved))


def test_arena_psfg_v1_boot_crash():
    """PSFG v1: crashed at boot (SyntaxError in a bundle patch); no arena progress ever."""
    bad = (FIX / "psfg-bootstrap.log").read_text()
    out = ops_watch.backtest({"run": "psfg-v1", "launched": 0, "boot_s": 900, "cadence_s": 60,
                              "targets": {"infra.running": 40}}, [], log_text=bad, until=1500, live_until=10**9)
    got = rules(out)
    assert "crash" in got and "boot" in got
    assert min(m for m, a in out if a.rule == "crash") == 0.0


def test_arena_arc2_snapshot_mapping_and_failures():
    from arc_infra.runops import arena

    snap = json.loads((FIX / "arc2-hx-two-snapshot.json").read_text())
    a = arena.arc2_row(snap, 1000.0)
    assert a["m"]["infra.running"] == 35 and a["m"]["play.slots"] == 49 and 1400 < a["m"]["infra.gen_tok_s"] < 1500
    b = arena.arc2_row({**snap, "completion_tokens": snap["completion_tokens"] + 90_000, "failures": 3}, 1060.0,
                       prev=(1000.0, snap["completion_tokens"]))
    assert b["m"]["infra.gen_tok_s"] == 1500 and b["m"]["play.errors"] == 3
    w = Watch({"run": "arc2", "launched": 0, "boot_s": 600})
    assert [x.rule for x in w.check(Snapshot(now=1060, rows=[{**a, "t": 300}, b], live=True))] == ["failures"]


PSFG2_LAUNCH = 1792830697.0  # psfgab v2b, 08:31:37 UTC
PSFG2_HANG = 1792834350.3  # 09:32:30: the engine hangs; the trial loop keeps writing rows; restart at 09:41


def test_psfg2_engine_hang_fires_frozen_within_6_min():
    """24 Oct: watch.py stayed quiet through a 9-min engine hang (rows kept coming, the copied rate field froze at a
    healthy 1586.3). The token counter catches it, via the adapter and via a feed that only copies the rate field."""
    from arc_infra.runops import arena

    r = arena.progress_rows(rows("psfgab-v2b-stall-progress.jsonl"))
    plan = {"run": "psfg2", "launched": PSFG2_LAUNCH, "boot_s": 1500, "cadence_s": 15,
            "targets": {"infra.gen_tok_s": {"plan": 1500, "steady": True}, "infra.running": 40}}
    out = ops_watch.backtest(plan, r, live_until=10**10, until=r[-1]["t"])
    hit = [m for m, a in out if a.rule == "frozen:infra.gen_tokens"]
    assert hit and 0 < hit[0] - (PSFG2_HANG - PSFG2_LAUNCH) / 60 <= 6
    copied = [{"t": x["t"], "m": {"infra.gen_tok_s": p["live"]["engine"]["tokens_per_second_recent"],
                                  "infra.running": p["live"]["engine"]["running"]}}
              for x, p in zip(r, rows("psfgab-v2b-stall-progress.jsonl"))]
    out = ops_watch.backtest(plan, copied, live_until=10**10, until=r[-1]["t"])
    hit = [m for m, a in out if a.rule == "frozen:infra.gen_tok_s"]
    assert hit and 0 < hit[0] - (PSFG2_HANG - PSFG2_LAUNCH) / 60 <= 6
    assert "rate:infra.gen_tok_s" not in rules(out)  # why the first watcher missed it


def test_frozen_quiet_when_drained_or_silent():
    w = Watch({"run": "x", "launched": 0, "boot_s": 60, "cadence_s": 60})
    flat = [{"t": 60.0 * i, "m": {"infra.gen_tokens": 5.0 if i > 5 else float(i), "infra.running": 0.0}}
            for i in range(30)]
    assert w.check(Snapshot(now=1740, rows=flat)) == []  # nothing in flight: a drained run
    busy = [{**x, "m": {**x["m"], "infra.running": 3.0}} for x in flat]
    assert [a.rule for a in w.check(Snapshot(now=1740, rows=busy))] == ["frozen:infra.gen_tokens"]
    w2 = Watch({"run": "x", "launched": 0, "boot_s": 60, "cadence_s": 60})
    assert "frozen:infra.gen_tokens" not in [a.rule for a in w2.check(Snapshot(now=1740 + 900, rows=busy))]


def test_watch_state_round_trip():
    w = Watch({"run": "x", "launched": 0, "boot_s": 60})
    r = [{"t": 100.0, "m": {}, "alert": "run says hi"}]
    assert [a.rule for a in w.check(Snapshot(now=200, rows=r))] == ["run-alert"]
    w2 = Watch({"run": "x", "launched": 0, "boot_s": 60})
    w2.restore(json.loads(json.dumps(w.state())))
    assert w2.check(Snapshot(now=260, rows=r)) == []


def test_watcher_survives_sigterm_and_failing_pass_then_beats(tmp_path, monkeypatch):
    """24 Oct: an arm's `pkill -f "tools/ops/watch.py run"` killed every arm's watcher mid-stall."""
    import os
    import signal
    import types

    monkeypatch.setattr(ops_watch, "WDIR", tmp_path)
    monkeypatch.setattr(ops_watch.paths, "SCRATCH", tmp_path)
    calls = []

    def snap(plan, metrics):
        calls.append(1)
        if len(calls) == 1:
            os.kill(os.getpid(), signal.SIGTERM)  # ignored
            raise OSError("rclone timed out")  # reported, retried
        return Snapshot(now=1e9, rows=[{"t": 1e9 - 30, "m": {}, "alert": "hello"}], ended=len(calls) >= 3,
                        clean_end=True)

    monkeypatch.setattr(ops_watch, "snapshot", snap)
    plan = tmp_path / "plan.json"
    plan.write_text(json.dumps({"run": "r1", "arm": "rl", "launched": 0, "boot_s": 60}))
    saved = {s: signal.getsignal(s) for s in (signal.SIGTERM, signal.SIGHUP)}
    try:
        assert ops_watch.cmd_run(types.SimpleNamespace(plan=str(plan), interval=0.01)) == 0
    finally:
        for s, h in saved.items():
            signal.signal(s, h)
    assert len(calls) == 3 and json.loads((tmp_path / "r1.beat.json").read_text())["ended"]
    assert json.loads((tmp_path / "r1.state.json").read_text())["seen"]  # the run alert, not re-fired after a restart
    (tmp_path / "r2.beat.json").write_text(json.dumps({"run": "r2", "arm": "pth", "pid": 2**22 + 7, "t": 1000,
                                                       "interval": 120, "ended": False}))
    assert ops_watch.stale_beats(now=1000 + 601, wdir=tmp_path, live_runs={"r2"}) == [("r2", "pth", 601, False)]
    assert ops_watch.stale_beats(now=1000 + 500, wdir=tmp_path, live_runs={"r2"}) == []
    assert not (tmp_path / "r1.beat.json").exists()  # r1's watcher ended: its beat file is gone
    assert ops_watch.stale_beats(now=1000 + 601, wdir=tmp_path, live_runs=set()) == []  # r2 settled meanwhile
    assert not (tmp_path / "r2.beat.json").exists()


def test_c7f_nccl_hang_fires_progress_stall():
    """24 Oct: c7f hung in NCCL after step 15 (11:00:01) at 98-100% GPU util; nothing fired until the app died at ~11:11.
    One measured step interval (190 s) x 2.5 -> the alert ~8 min after the last step, before the death."""
    r = [x for x in rows("opdS-c7f.jsonl") if not str(x.get("alert", "")).startswith("watch:")]
    plan = {"run": "c7f", "launched": t0(r), "boot_s": 600, "cadence_s": 60, "progress_key": "train.step"}
    death = 1792840260.0  # ~11:11 UTC
    out = ops_watch.backtest(plan, r, live_until=death, until=death)
    hit = [m for m, a in out if a.rule == "progress:train.step"]
    last_step = 1792839601.7
    assert hit and 0 < (t0(r) + hit[0] * 60 - last_step) / 60 <= 9
    healthy = [{"t": 1000.0 + 190 * i, "m": {"train.step": float(i)}} for i in range(20)]
    assert "progress:train.step" not in rules(ops_watch.backtest({**plan, "launched": 1000}, healthy,
                                                                 live_until=10**9, until=1000 + 190 * 19 + 300))


def test_follow_record_takes_the_current_attempt_and_resets_rules(tmp_path, monkeypatch, capsys):
    """One plan and one watcher across relaunches (records.py reserve of an ended run = attempt N+1)."""
    import types

    from arc_infra.runops import records

    monkeypatch.setattr(ops_watch, "WDIR", tmp_path)
    monkeypatch.setattr(ops_watch.paths, "SCRATCH", tmp_path)
    rec = {"run": "r3", "attempt": 1, "ledger": [{"row": {"start_epoch": 1000, "app_id": "ap-a1"}}]}
    monkeypatch.setattr(records, "get", lambda run, root=None: rec)
    seen = []

    def snap(plan, metrics):
        seen.append((plan["app_id"], plan["launched"]))
        if len(seen) == 2:  # attempt 1 died; the arm reserved the same run again
            rec.update(attempt=2, ledger=rec["ledger"] + [{"row": {"start_epoch": 5000, "app_id": "ap-b2"}}])
        return Snapshot(now=1000 + 4000 * len(seen), rows=[{"t": 1010.0, "m": {}, "alert": "old attempt"}],
                        ended=len(seen) >= 4, clean_end=True)

    monkeypatch.setattr(ops_watch, "snapshot", snap)
    plan = tmp_path / "plan.json"
    plan.write_text(json.dumps({"run": "r3", "arm": "rl", "platform": "modal", "boot_s": 60, "follow_record": True}))
    assert ops_watch.cmd_run(types.SimpleNamespace(plan=str(plan), interval=0.01)) == 0
    assert seen[:2] == [("ap-a1", 1000), ("ap-a1", 1000)] and seen[2:] == [("ap-b2", 5000)] * 2
    out = capsys.readouterr().out
    assert "attempt 2 (was 1): rule state reset" in out and out.count("old attempt") == 1  # attempt 2 drops its rows
    assert json.loads((tmp_path / "r3.state.json").read_text())["attempt"] == 2
    with pytest.raises(ValueError):
        validate_plan({"run": "x", "boot_s": 60, "platform": "modal"})  # without follow_record: launched + app_id


def test_expected_read_missing_fires_once():
    """v6e (24 Oct) ran without its step-0 base read: no anchor for the held-out ratio."""
    plan = {"run": "x", "launched": 0, "boot_s": 60, "play_key": "infra.running",
            "expect": {"eval.heldout_rhae10k": 1800}}
    r = [{"t": float(t), "m": {"infra.running": 40.0}} for t in range(60, 4000, 60)]
    out = ops_watch.backtest(plan, r, live_until=10**9, until=4000)
    hits = [m for m, a in out if a.rule == "expect:eval.heldout_rhae10k"]
    assert len(hits) == 1 and 31 <= hits[0] <= 33
    r2 = r + [{"t": 1500.0, "m": {"eval.heldout_rhae10k": 0.2}}]
    assert not [a for _, a in ops_watch.backtest(plan, r2, live_until=10**9, until=4000) if a.rule.startswith("expect")]


def test_starve_clocked_on_server_ready_fires_on_v6_v6b_quiet_on_v6d():
    """24 Oct: hero-v6/v6b starved (2-5 live tries vs 38) ~10 min before plan@10m; v6d ramped through half plan."""
    def first_starve(name, plan):
        r = [x for x in rows(name) if not str(x.get("alert", "")).startswith("watch:")]
        gen = sorted((x["t"], x["m"]["infra.gen_tok_s"]) for x in r if "infra.gen_tok_s" in x.get("m", {}))
        ready = next(t for t, v in gen if v > 0.05 * plan["targets"]["infra.gen_tok_s"]["plan"])
        p = {"run": name, "launched": t0(r), **plan}
        hit = [m for m, a in ops_watch.backtest(p, r) if a.rule == "starve"]
        return (t0(r) + hit[0] * 60 - ready) / 60 if hit else None

    assert 5 <= first_starve("hero-v6.jsonl", RL_HERO) <= 7
    assert 5 <= first_starve("hero-v6b.jsonl", RL_HERO) <= 7
    v6d = {**RL_HERO, "targets": {**RL_HERO["targets"], "infra.gen_tok_s": {"plan": 1870, "steady": True}},
           "play_key": "infra.running"}
    assert first_starve("hero-v6d.jsonl", v6d) is None
