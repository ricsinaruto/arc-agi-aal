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


def test_arena_arc2_counts_extra_runs_tokens():
    """arc2 twin (28 Oct): once the entries have ended, second tries (``extra_tokens``) keep the server busy while the
    read-out's completion_tokens stays flat; the counter and the rate include them."""
    from arc_infra.runops import arena

    snap = {**json.loads((FIX / "arc2-hx-two-snapshot.json").read_text()), "extra_tokens": 0}
    later = {**snap, "extra_tokens": 60_000}
    row = arena.arc2_row(later, 1060.0, prev=(1000.0, arena.arc2_tokens(snap)))
    assert row["m"]["infra.gen_tokens"] == snap["completion_tokens"] + 60_000 and row["m"]["infra.gen_tok_s"] == 1000
    assert arena.arc2_tokens({"completion_tokens": 5}) == 5 and arena.arc2_tokens({}) is None


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


def test_frozen_counts_only_the_busy_span():
    """v17 29 Oct: a vLLM worker hang (10 running, counter flat) fires at 120 s; v15: a play start after a 21-min
    train phase (counter flat through it, running 0 -> 2) stays quiet."""
    plan = {"run": "x", "launched": 0, "boot_s": 60, "cadence_s": 65, "counters": {"infra.gen_tokens": 120}}
    train = [{"t": 65.0 * i, "m": {"infra.gen_tokens": 100.0, "infra.running": 0.0}} for i in range(20)]
    start = train + [{"t": 65.0 * 20, "m": {"infra.gen_tokens": 100.0, "infra.running": 2.0}}]
    assert "frozen:infra.gen_tokens" not in [a.rule for a in Watch(plan).check(Snapshot(now=65 * 20, rows=start))]
    hung = start + [{"t": 65.0 * i, "m": {"infra.gen_tokens": 100.0, "infra.running": 10.0}} for i in (21, 22)]
    assert "frozen:infra.gen_tokens" in [a.rule for a in Watch(plan).check(Snapshot(now=65 * 22, rows=hung))]


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


def test_resumed_run_ignores_the_earlier_attempts_rows():
    """sft-p75-qad-v2 27 Oct: attempt 2 resumed 6 h after attempt 1 stopped at step 129, appending to the same metrics
    file; its boot must not read attempt 1's last step or row (stuck / silent), and a stall of attempt 2 still fires."""
    old = [{"t": 1000.0 + 100 * i, "m": {"train.step": float(i)}} for i in range(130)]
    launched = old[-1]["t"] + 6 * 3600
    new = [{"t": launched + 600 + 100 * i, "m": {"train.step": 130.0 + i}} for i in range(10)]
    plan = {"run": "v2", "launched": launched, "boot_s": 900, "cadence_s": 15, "progress_key": "train.step",
            "progress_k": 8, "silence_s": 900}
    boot = ops_watch.backtest(plan, old + new, live_until=10**10, remote_until=launched + 600 + 900,
                              until=launched + 600 + 900)
    assert boot == []
    stall = ops_watch.backtest(plan, old + new, live_until=10**10, until=new[-1]["t"] + 30 * 60)
    assert "progress:train.step" in rules(stall)


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


def test_follow_record_paths_take_the_attempt_folder():
    """v6k 25 Oct: attempt 2 writes validation/<run>-attempt2/; a watcher reading validation/<run>/ saw nothing."""
    plan = {"run": "rl1-x", "follow_record": True, "launched": 0,
            "feed": "modal:arc-prolong-runs:validation/rl1-x/arena/trials/rl-loop/rl/feed.jsonl",
            "logs": ["modal:arc-prolong-runs:{out}/log.txt", "nebius:runs/rl1-x/final.json"]}
    rec = {"run": "rl1-x", "attempt": 2, "attempts": [{"n": 1}, {"n": 2}],
           "ledger": [{"row": {"start_epoch": 5, "app_id": "ap-2"}}]}
    eff, n = ops_watch.follow(plan, rec)
    assert n == 2 and eff["app_id"] == "ap-2" and eff["launched"] == 5
    assert eff["feed"] == "modal:arc-prolong-runs:validation/rl1-x-attempt2/arena/trials/rl-loop/rl/feed.jsonl"
    assert eff["logs"] == ["modal:arc-prolong-runs:rl1-x-attempt2/log.txt", "nebius:runs/rl1-x/final.json"]
    eff1, _ = ops_watch.follow(plan, {**rec, "attempt": 1, "attempts": [{"n": 1}]})
    assert eff1["feed"] == plan["feed"]


def test_level_bench_rows_rate_and_stamps():
    from arc_infra.runops.arena import level_bench_rows

    rows = [{"event": "level_bench_start"},
            {"event": "level_bench_progress", "elapsed_seconds": 10.0, "completion_tokens": 0, "running": 4, "failed": 0,
             "done": 0, "clears": 0, "gate": {"inflight": 4, "waiting": 1}},
            {"event": "level_bench_progress", "elapsed_seconds": 40.0, "completion_tokens": 3000, "running": 4,
             "failed": 1, "done": 2, "clears": 1, "gate": {"inflight": 3, "waiting": 0}}]
    out = level_bench_rows(rows, 1000.0)
    assert [r["t"] for r in out] == [970.0, 1000.0]
    assert out[1]["m"]["infra.gen_tok_s"] == 100.0 and out[1]["m"]["infra.gen_tokens"] == 3000
    assert out[1]["m"]["play.errors"] == 1 and out[0]["m"]["infra.waiting"] == 1
    assert level_bench_rows(rows, None) == [] and level_bench_rows([], 5.0) == []


def _replay(monkeypatch, tmp_path, r, plan, end):
    """cmd_run over a recorded log: one pass per minute from launch to ``end`` (clean end), as backtest replays it."""
    import types

    monkeypatch.setattr(ops_watch, "WDIR", tmp_path)
    monkeypatch.setattr(ops_watch.paths, "SCRATCH", tmp_path)
    clock = {"now": plan["launched"]}

    def snap(_plan, _metrics):
        clock["now"] += 60
        return Snapshot(now=clock["now"], rows=r, ended=clock["now"] >= end, clean_end=True)

    monkeypatch.setattr(ops_watch, "snapshot", snap)
    path = tmp_path / "plan.json"
    path.write_text(json.dumps(plan))
    run = lambda: ops_watch.cmd_run(types.SimpleNamespace(plan=str(path), interval=0.0, exit_on_alert=True))
    return run, clock


def test_exit_on_alert_wakes_at_the_first_alert_and_restart_does_not_refire(tmp_path, monkeypatch, capsys):
    """E1 ran ~85 min past its stop rule because the watcher printed its ALERT and kept running, so
    the owning agent's background task never woke. hero-v6 (starved play): exit 4 at the first alert (starve, 14 min);
    the restarted watcher keeps rule state and exits at the next new alert, never on the handled one."""
    r = rows("hero-v6.jsonl")
    plan = {"run": "hero-v6", "arm": "rl", "launched": t0(r), **RL_HERO}
    run, clock = _replay(monkeypatch, tmp_path, r, plan, t0(r) + 3600)
    assert run() == 4 and (clock["now"] - t0(r)) / 60 == 14
    out = capsys.readouterr().out
    assert "[starve]" in out and "WOKE hero-v6" in out and "ALERT" in out.split("WOKE")[0]
    assert json.loads((tmp_path / "hero-v6.beat.json").read_text())["woke"]
    assert run() == 4  # restarted after handling: the next alert (plan@10m at 20 min), not starve again
    out = capsys.readouterr().out
    assert "[plan@10m]" in out and "[starve]" not in out


def test_exit_on_alert_healthy_log_runs_to_the_end(tmp_path, monkeypatch, capsys):
    r = rows("ovf3-80m.jsonl")
    plan = {"run": "ovf3", "arm": "rl", "launched": t0(r), "boot_s": 1260, "cadence_s": 60,
            "targets": {"infra.running": 40, "infra.gen_tok_s": 1000, "infra.gpu_util.*": 85, "infra.vram_gib.*": 250}}
    end = max(x["t"] for x in r)
    run, clock = _replay(monkeypatch, tmp_path, r, plan, end)
    assert run() == 0 and clock["now"] >= end
    out = capsys.readouterr().out
    assert "ALERT" not in out and "END ovf3" in out


SAMPLER_PLAN = {"boot_s": 900, "cadence_s": 60, "play_key": "infra.gen_tok_s",
                "targets": {"infra.gen_tok_s": {"plan": 1000, "steady": True}, "infra.gpu_util.*": 85}}


def test_sampler_rows_map_the_runops_sampler():
    """RL redesign runs (26 Oct) write only live/sampler.jsonl: relative t, per-card util/mem, vLLM gauges/counters."""
    from arc_infra.runops import arena

    r = arena.sampler_rows(rows("sampler-t3-v3.jsonl"), launched_epoch=1000.0)
    assert r[0]["t"] == 1065.0 and r[0]["m"] == {"infra.gpu_util.0": 0.0, "infra.vram_gib.0": 0.0}
    m = r[-1]["m"]
    assert m["infra.gen_tokens"] == 582475.0 and m["infra.vram_gib.0"] == 215.1 and 0 < m["infra.kv_frac"] < 1
    assert abs(m["infra.gen_tok_s"] - (582475 - 467113) / 62) < 1e-6 and m["play.preemptions"] == 0
    assert arena.sampler_rows(rows("sampler-t3-v3.jsonl"), mtime=5000.0)[-1]["t"] == 5000.0  # file time wins
    two = arena.sampler_rows(rows("sampler-hero-v6m.jsonl"), 0.0)[-1]["m"]
    assert {"infra.gpu_util.1", "infra.vram_gib.1"} <= set(two)
    host = arena.cpulog_rows((FIX / "cpulog-t3-v3.txt").read_text())
    assert host and host[-1]["m"]["infra.host_ram_gib"] > 300 and all(b["t"] - a["t"] >= 30 for a, b in zip(host, host[1:]))


def test_sampler_backtest_dead_t3_fires_healthy_quiet():
    """rl1-loop-seq-t3-v1-attempt2 died after boot (util 0, 1,463 tokens, rows stop at 13 min): unclean end, silence
    and the plan check fire; t3-v3 and hero-v6m (two cards, card 1 the trainer) stay quiet."""
    from arc_infra.runops import arena

    L = 1_793_000_000.0
    dead = arena.sampler_rows(rows("sampler-t3-v1a2.jsonl"), L)
    out = ops_watch.backtest({"run": "t3a2", "launched": L, **SAMPLER_PLAN}, dead, live_until=dead[-1]["t"] + 600,
                             until=dead[-1]["t"] + 1200)
    assert {"unclean-end", "silent", "plan@10m"} <= set(rules(out))
    for name, extra in (("sampler-t3-v3.jsonl", {}), ("sampler-hero-v6m.jsonl", {"util_exempt": ["infra.gpu_util.1"],
                        "targets": {"infra.gen_tok_s": {"plan": 600, "steady": True}, "infra.gpu_util.*": 85}})):
        ok = arena.sampler_rows(rows(name), L)
        assert ops_watch.backtest({"run": name, "launched": L, **SAMPLER_PLAN, **extra}, ok,
                                  live_until=ok[-1]["t"] + 1800, until=ok[-1]["t"]) == []


def test_no_play_fires_two_passes_past_boot_while_rows_arrive():
    """t3-v1-attempt2 alerted only at its silence (24 min). no-play fires at boot + 2 passes (12 min
    with the measured 10-min boot); t3-v3 and hero-v6m stay quiet with their measured boots; off without a sampler."""
    from arc_infra.runops import arena

    L = 1_793_000_000.0
    plan = {**SAMPLER_PLAN, "boot_s": 600, "arena": {"sampler": "modal:v:p"}, "launched": L}

    def nplay(name, **extra):
        r = arena.sampler_rows(rows(name), L)
        out = ops_watch.backtest({**plan, "run": name, **extra}, r, live_until=r[-1]["t"] + 600, until=r[-1]["t"] + 60)
        return [m for m, a in out if a.rule == "no-play"]

    assert nplay("sampler-t3-v1a2.jsonl") == [12.0]
    assert nplay("sampler-t3-v3.jsonl") == []
    v6m = {"util_exempt": ["infra.gpu_util.1"], "targets": {"infra.gen_tok_s": {"plan": 600, "steady": True}}}
    assert nplay("sampler-hero-v6m.jsonl", **v6m) == []
    assert nplay("sampler-hero-v6m.jsonl", **v6m, boot_s=300) == [7.0]  # boot_s under the real boot: it fires
    assert nplay("sampler-t3-v1a2.jsonl", no_play=False) == [] and nplay("sampler-t3-v1a2.jsonl", arena={}) == []


T3_LAUNCH = 1793010890.0  # rl1-loop-seq-t3-v3: its cpulog's first epoch (the sampler's t = 0)


def _t3(cut=None):
    """t3-v3's sampler + feed. The feed's seq_train rows are reconstructed (play start + play_seconds), as rl_seq
    writes them from b0b009ca0. From 10:55:56 the run looped on empty failed groups (a play phase every ~10 s)."""
    from arc_infra.runops import arena

    r = arena.sampler_rows(rows("sampler-t3-v3-full.jsonl"), T3_LAUNCH) + arena.seq_feed_rows(rows("feed-t3-v3.jsonl"))
    return [x for x in r if cut is None or x["t"] <= cut]


T3_PLAN = {"run": "t3v3", "launched": T3_LAUNCH, "boot_s": 600, "cadence_s": 60, "play_key": "infra.gen_tok_s",
           "arena": {"sampler": "modal:v:p", "feed": "modal:v:f"}, "train_max_s": 1200,
           "targets": {"infra.gen_tok_s": {"plan": 1000, "steady": True}, "infra.gpu_util.*": 85}}


def test_phase_aware_train_phase_quiet_failure_loop_fires():
    """every rl_seq train phase sleeps the servers for minutes; no-play/starve must not fire on it.
    t3-v3's first train phase (10:52:49-10:55:56, gen flat, util 0) stays quiet; the failure loop after it fires
    phase-churn at 10:56."""
    r = _t3()
    end = max(x["t"] for x in r)
    out = ops_watch.backtest(T3_PLAN, r, live_until=end + 600, until=end + 120)
    assert [(m, a.rule) for m, a in out] == [(22.0, "phase-churn")]
    healthy = _t3(cut=1793012186.0)
    assert ops_watch.backtest(T3_PLAN, healthy, live_until=1793015000.0, until=1793012186.0) == []
    blind = [x for x in r if not any(k.startswith("seq.") for k in x["m"])]  # the same rows without the feed:
    out = ops_watch.backtest(T3_PLAN, blind, live_until=end + 600, until=end + 120)  # no-play inside the train phase
    assert (21.0, "no-play") in [(m, a.rule) for m, a in out]


def test_train_cold_fires_at_15_min_past_a_long_train_row_s():
    """A long train_row_s (the old pass has no progress row) must not stretch the 15-min B300 rule; a pass that is busy
    on average stays quiet though many samples read under 30%."""
    L, plan = 1000.0, {"run": "tr", "launched": 1000.0, "boot_s": 60, "cadence_s": 60, "train_max_s": 10**6,
                       "train_row_s": 2400}
    feed = [{"t": L + 100, "m": {"seq.phase": 1.0}}, {"t": L + 400, "m": {"seq.phase": 0.0}},
            {"t": L + 500, "m": {"seq.train_rows": 1.0}}]
    at = lambda i: L + 500 + 60 * i  # noqa: E731
    cold = [{"t": at(i), "m": {"infra.gpu_util.0": 2.0}} for i in range(30)]
    out = ops_watch.backtest(plan, feed + cold, live_until=10**10, until=at(29))
    assert rules(out) == ["train-cold"] and (min(m for m, a in out) * 60 - 500) <= 960
    # a10's old pass: 7 of 10 samples at 100, 3 at 0: a 70% mean, and two cold samples in a row happen
    busy = [{"t": at(i), "m": {"infra.gpu_util.0": 0.0 if i % 10 in (3, 4, 8) else 100.0}} for i in range(30)]
    assert ops_watch.backtest(plan, feed + busy, live_until=10**10, until=at(29)) == []
    # progress rows inside the span: not a hang
    rows = [{"t": at(i), "m": {"seq.train_rows": float(i + 2)}} for i in range(0, 30, 5)]
    assert ops_watch.backtest(plan, feed + cold + rows, live_until=10**10, until=at(29)) == []
    # the trainer's model load keeps train_load_s
    load = [r for r in feed if "seq.train_rows" not in r["m"]]
    assert "train-cold" not in rules(ops_watch.backtest(plan, load + cold, live_until=10**10, until=at(29)))


def test_train_long_and_train_idle():
    """A running train phase: train-long past train_max_s (2x the longest measured phase once one ended); train-idle
    needs both no progress row for train_row_s and the cards under 30%."""
    L, plan = 1000.0, {"run": "tr", "launched": 1000.0, "boot_s": 60, "cadence_s": 60, "train_max_s": 900}
    feed = [{"t": L + 100, "m": {"seq.phase": 1.0}}, {"t": L + 400, "m": {"seq.phase": 0.0}},
            {"t": L + 500, "m": {"seq.train_rows": 1.0}}]
    hot = [{"t": L + 400 + 60 * i, "m": {"infra.gpu_util.0": 95.0}} for i in range(30)]
    out = ops_watch.backtest(plan, feed + hot, live_until=10**10, until=L + 400 + 60 * 29)
    assert rules(out) == ["train-long"] and min(m for m, a in out) * 60 >= 400 + 900
    cold = [{"t": L + 400 + 60 * i, "m": {"infra.gpu_util.0": 2.0}} for i in range(30)]
    out = ops_watch.backtest({**plan, "train_max_s": 10**6}, feed + cold, live_until=10**10, until=L + 400 + 60 * 29)
    assert {"train-idle", "train-cold"} == set(rules(out)) and min(m for m, a in out) * 60 >= 500 + 600
    # the first train phase before any progress row is the trainer's model load: train_load_s (1500), not train_row_s
    load = [r for r in feed if "seq.train_rows" not in r["m"]]
    out = ops_watch.backtest({**plan, "train_max_s": 10**6}, load + cold, live_until=10**10, until=L + 400 + 60 * 29)
    assert rules(out) == ["train-idle"] and min(m for m, a in out) * 60 >= 400 + 1500
    # v15 29 Oct 13:26: one 0-util sample exported twice between hot ones is not two cold reads
    blip = hot[:12] + [{"t": L + 400 + 60 * 12 + 1, "m": {"infra.gpu_util.0": 0.0}}] * 2
    out = ops_watch.backtest({**plan, "train_max_s": 10**6}, feed + blip, live_until=10**10, until=L + 400 + 60 * 12 + 30)
    assert rules(out) == []
    done = feed + [{"t": L + 700, "m": {"seq.phase": 1.0}}, {"t": L + 1000, "m": {"seq.phase": 0.0}}]
    w = ops_watch.Watch({**plan, "train_max_s": 10**6})  # measured: 300 s phase -> ceiling 600 s
    assert [a.rule for a in w.check(Snapshot(now=L + 1000 + 601, rows=done + hot))] == ["train-long"]


def test_follow_record_substitutes_inside_the_arena_block():
    rec = {"run": "r5", "attempt": 2, "ledger": [{"row": {"start_epoch": 9, "app_id": "ap-x"}}]}
    import types
    from arc_infra.runops import records
    orig = records.out_id
    records.out_id = lambda run, rec=None: "r5-attempt2"
    try:
        eff, _ = ops_watch.follow({"run": "r5", "follow_record": True,
                                   "arena": {"sampler": "modal:v:validation/r5/live/sampler.jsonl"}}, rec)
    finally:
        records.out_id = orig
    assert eff["arena"]["sampler"] == "modal:v:validation/r5-attempt2/live/sampler.jsonl"


def _two_servers(idle_from=None, idle_to=None, hang_from=None, n=20, L=1_793_000_000.0):
    """A fake 2-card sampler (H2 26 Oct): :1234 on card 0, :1236 on card 1, rows every 60 s from t = 600 (play).
    :1236 idles (0 running, no tokens, card 1 at 0%) over [idle_from, idle_to) or hangs (running, tokens flat, card
    busy) from hang_from."""
    out, gen = [], {"1234": 0.0, "1236": 0.0}
    for i in range(n):
        t = 600 + 60 * i
        idle = idle_from is not None and idle_from <= t < (idle_to or 1e9)
        hang = hang_from is not None and t >= hang_from
        gen["1234"] += 60 * 1500 + 37 * (i % 3)  # a live rate never repeats one exact value (the frozen rule's)
        gen["1236"] += 0 if idle or hang else 60 * 1500 + 41 * (i % 2)
        per = {"1234": {"num_requests_running": 60.0, "num_requests_waiting": 0.0, "kv_cache_usage_perc": 0.3,
                        "generation_tokens_total": gen["1234"], "num_preemptions_total": 0.0},
               "1236": {"num_requests_running": 0.0 if idle else 60.0, "num_requests_waiting": 0.0,
                        "kv_cache_usage_perc": 0.0 if idle else 0.3, "generation_tokens_total": gen["1236"],
                        "num_preemptions_total": 0.0}}
        tot = {k: sum(p[k] for p in per.values()) for k in per["1234"]}
        tot["kv_cache_usage_perc"] /= 2
        out.append({"t": t, "util": [98.0, 0.0 if idle else 97.0], "mem_gib": [212.0, 212.0], "vllm": tot,
                    "vllm_servers": per})
    return out, L


TWO_PLAN = {"boot_s": 600, "cadence_s": 60, "arena": {"sampler": "modal:v:p"}, "play_key": "infra.gen_tok_s",
            "live_key": "infra.running", "targets": {"infra.running": 120, "infra.gen_tok_s": 3000}}


def test_sampler_rows_split_per_server():
    from arc_infra.runops import arena

    raw, L = _two_servers()
    m = arena.sampler_rows(raw, L)[-1]["m"]
    assert m["infra.running.s1234"] == 60 and m["infra.running.s1236"] == 60 and m["infra.running"] == 120
    assert 1490 < m["infra.gen_tok_s.s1236"] < 1510 and 2990 < m["infra.gen_tok_s"] < 3010
    assert m["infra.kv_frac.s1236"] == 0.3
    assert arena.server_card(1234) == 0 and arena.server_card("1236") == 1 and arena.server_card(1238) == 3


def test_sampler_rows_prefix_hit_per_server():
    from arc_infra.runops import arena

    srv = lambda h1, q1, h2, q2: {"1234": {"prefix_cache_hits_total": h1, "prefix_cache_queries_total": q1},
                                  "1236": {"prefix_cache_hits_total": h2, "prefix_cache_queries_total": q2}}
    raw = [{"t": 0, "epoch": 1000, "vllm_servers": srv(0, 0, 0, 0)}, {"t": 60, "epoch": 1060, "vllm_servers": srv(90, 100, 80, 100)},
           {"t": 120, "epoch": 1120, "vllm_servers": srv(90, 100, 80, 100)}]  # no new queries: no rate
    rows = arena.sampler_rows(raw)
    assert "infra.prefix_hit.s1234" not in rows[0]["m"]
    assert rows[1]["m"]["infra.prefix_hit.s1234"] == 0.9 and rows[1]["m"]["infra.prefix_hit.s1236"] == 0.8
    assert "infra.prefix_hit.s1234" not in rows[2]["m"]


def test_one_idle_server_fires_per_server_rules_healthy_stays_quiet():
    """H2: card 1's server idle while :1234 played (the totals at half plan hide it from no-play and starve)."""
    from arc_infra.runops import arena

    def fired(**kw):
        raw, L = _two_servers(**kw)
        r = arena.sampler_rows(raw, L)
        out = ops_watch.backtest({**TWO_PLAN, "run": "h2", "launched": L}, r, live_until=r[-1]["t"] + 600,
                                 until=r[-1]["t"])
        return {a.rule for _, a in out}

    assert fired() == set()
    idle = fired(idle_from=600, idle_to=1500)
    assert {"no-play:1236", "starve:1236"} <= idle, idle
    assert not idle & {"no-play", "starve", "no-play:1234", "starve:1234"}
    short = fired(idle_from=900, idle_to=960)  # one quiet row (a switch's tail): not two passes
    assert not any(rule.startswith("no-play") for rule in short), short
    hung = fired(hang_from=900)
    assert "frozen:infra.gen_tokens.s1236" in hung, hung
    assert not any(rule.startswith("no-play") for rule in hung)  # card busy: the hang rule's, not idle's


def test_per_server_rules_are_off_on_one_server():
    from arc_infra.runops import arena

    r = arena.sampler_rows(rows("sampler-t3-v3.jsonl"), 1_793_000_000.0)
    assert not any(".s1" in k for x in r for k in x["m"])
    assert Watch({"run": "x", "launched": 0, "boot_s": 1})._servers(r) == []


def test_nebius_late_metrics_sync_is_not_a_stall_or_silence():
    """26 Oct b2-rl-hspeed-a-v2 (Nebius read): train-metrics.jsonl (a row every 30 s) reaches the bucket every ~4 min
    while bootstrap.log is current; the progress rule fired "gen_tokens stuck" between syncs. Replayed with 4-min
    syncs: quiet with the Nebius lag allowance, firing without it; a fresh run log vetoes the silence rule."""
    r = rows("hspeed-a-v2.jsonl")
    plan = {"run": "hspeed", "platform": "nebius", "metrics_remote": "nebius:x", "launched": r[0]["t"] - 1800,
            "boot_s": 1800, "cadence_s": 300, "targets": {"infra.gen_tok_s": {"plan": 720, "steady": True}},
            "progress_key": "infra.gen_tokens"}

    def replay(p):
        w, fired, t0 = ops_watch.Watch(p), [], r[0]["t"]
        for i in range(int((r[-1]["t"] - t0) / 60) + 5):
            now = t0 + 60 * i
            synced = t0 + 240 * ((now - t0) // 240)  # the bucket copy as of the last 4-min sync
            fired += [a.rule for a in w.check(Snapshot(now=now, rows=[x for x in r if x["t"] <= synced], live=True))]
        return fired

    assert replay(plan) == []
    assert "progress:infra.gen_tokens" in replay({**plan, "metrics_lag_s": 0})
    w = ops_watch.Watch({**plan, "silence_s": 600})
    stale = Snapshot(now=r[-1]["t"] + 1200, rows=r, live=True, remote_age=1200)
    assert "silent" not in [a.rule for a in w.check(Snapshot(**{**stale.__dict__, "log_age": 30}))]
    assert "silent" in [a.rule for a in ops_watch.Watch({**plan, "silence_s": 600}).check(stale)]


def test_plan_marks_count_play_time_not_train_phases():
    """H4 26 Oct: a 298 s play phase, then a train phase with the servers asleep: plan@10m read 13 s of play tail plus
    5 min of train phase and fired at 0. Marks now count play time; a slow second phase still fires."""
    W = ops_watch.Watch
    assert W._play_clock(0.0, [], 600) == 600
    assert W._play_clock(0.0, [(300.0, 1500.0)], 600) == 1800
    assert W._play_clock(0.0, [(300.0, None)], 600) is None
    assert W._play_clock(0.0, [(300.0, 1500.0)], 200) == 200
    L = 1000.0
    plan = {"run": "h4", "launched": L, "boot_s": 600, "cadence_s": 60, "play_key": "infra.running", "no_play": False,
            "targets": {"infra.running": {"plan": 220, "steady": True}}}

    def run(second_phase_running):
        rows = [{"t": L + 600, "m": {"seq.phase": 1.0}}, {"t": L + 900, "m": {"seq.phase": 0.0}},
                {"t": L + 2100, "m": {"seq.phase": 1.0}}]
        rows += [{"t": L + 600 + 60 * i, "m": {"infra.running": 230.0}} for i in range(5)]
        rows += [{"t": L + 900 + 60 * i, "m": {"infra.running": 0.0}} for i in range(20)]  # asleep
        rows += [{"t": L + 2100 + 60 * i, "m": {"infra.running": second_phase_running}} for i in range(20)]
        return [a.rule for m, a in ops_watch.backtest(plan, rows, live_until=10**10, until=L + 2100 + 60 * 19)]

    assert "plan@10m" not in run(225.0)
    assert "plan@10m" in run(40.0)


def test_confirm_needs_a_second_signal_for_rate_alerts():
    """arc2 conv1 27 Oct: lumpy reply-completion tokens read 801 tok/s at 95% util (plan@20m fired on one signal).
    With ``confirm`` the rate alerts need the listed key under its floor too."""
    L = 1000.0
    plan = {"run": "c1", "launched": L, "boot_s": 60, "cadence_s": 60, "no_play": False, "play_key": "infra.gen_tok_s",
            "targets": {"infra.gen_tok_s": {"plan": 1500, "steady": True}, "infra.gpu_util.*": 94},
            "confirm": {"infra.gen_tok_s": ["infra.gpu_util.*"]}}

    def run(util, with_confirm=True):
        p = plan if with_confirm else {k: v for k, v in plan.items() if k != "confirm"}
        rows = [{"t": L + 60 + 60 * i, "m": {"infra.gen_tok_s": 800.0, "infra.gpu_util.0": util,
                                              "infra.gen_tokens": 48000.0 * (i + 1)}} for i in range(45)]
        return {a.rule for m, a in ops_watch.backtest(p, rows, live_until=10**10, until=L + 60 * 45)}

    assert not {"plan@10m", "plan@20m", "rate:infra.gen_tok_s"} & run(95.0)
    assert {"plan@10m", "plan@20m", "rate:infra.gen_tok_s"} <= run(40.0)
    assert {"plan@10m", "plan@20m"} <= run(95.0, with_confirm=False)


def test_rate_floor_each_and_phase_window():
    """plan rate_floor 0.8; "each" judges every new per-phase value once; a steady target with
    phase_from_s is judged only from that far into the current play phase."""
    L = 1000.0
    plan = {"run": "e", "launched": L, "boot_s": 60, "cadence_s": 60, "rate_floor": 0.8,
            "targets": {"infra.running": 100, "infra.kv_frac": {"plan": 0.8, "steady": True, "phase_from_s": 600}},
            "each": {"seq.to_train_s": {"max": 10}, "train.call_util": {"min": 72}}}
    rows = [{"t": L + 60 * i, "m": {"infra.running": 100.0, "infra.gen_tok_s": 50.0, "infra.gen_tokens": 50.0 * i,
                                    "infra.kv_frac": 0.3 if i < 30 else 0.7}} for i in range(1, 60)]
    rows += [{"t": L + 600, "m": {"seq.to_train_s": 4.0}}, {"t": L + 900, "m": {"seq.to_train_s": 30.0}},
             {"t": L + 1000, "m": {"train.call_util": 65.0}}]
    w = ops_watch.Watch(plan)
    out = [a.rule for a in w.check(Snapshot(now=L + 1100, rows=rows, live=True))]
    assert out.count("each:seq.to_train_s") == 1 and "each:train.call_util" in out
    assert [a.rule for a in w.check(Snapshot(now=L + 1160, rows=rows, live=True))] == []  # judged once
    # kv 0.3 for the first 30 min, then 0.7: a 15-min median from launch + 20 min reads low; 0.7 < 0.8 * 0.8? no
    w2 = ops_watch.Watch({**plan, "each": {}})
    rules_at = lambda now: [a.rule for a in w2.check(Snapshot(now=now, rows=rows, live=True))]
    assert "rate:infra.kv_frac" in rules_at(L + 1500)  # no train phase: phase start = play start; 0.3 < 0.64
    assert "rate:infra.kv_frac" not in rules_at(L + 60 * 58)  # the window from play + 600 s now median 0.7 >= 0.64
    st = w.state()
    w3 = ops_watch.Watch(plan)
    w3.restore(st)
    assert [a.rule for a in w3.check(Snapshot(now=L + 1200, rows=rows, live=True))] == []  # restored: not again


def test_arena_rows_carry_the_per_phase_keys():
    from arc_infra.runops import arena

    feed = [{"epoch": 10.0, "kind": "seq_start", "setup_seconds": 78.2},
            {"epoch": 20.0, "kind": "seq_admit", "cpu": {"python:1392": 1.18, "python:1500": 0.6, "driver:9": 0.1}},
            {"epoch": 25.0, "kind": "seq_accumulate", "one_call": True, "call_util": 55.0, "setup_seconds": 50.0},  # cold
            {"epoch": 30.0, "kind": "seq_train"},
            {"epoch": 40.0, "kind": "seq_accumulate", "one_call": True, "call_util": 81.5, "setup_seconds": 3.1},
            {"epoch": 50.0, "kind": "seq_phase", "stop": {"switch_s": 9.0}, "start": {"switch_s": 25.0}}]
    m = {}
    for r in arena.seq_feed_rows(feed):
        m.update(r["m"])
    assert m["seq.setup_s"] == 78.2 and m["infra.api_cpu"] == 1.18 and m["train.call_util"] == 81.5
    assert m["train.setup_s"] == 3.1 and m["seq.to_train_s"] == 9.0 and m["seq.to_play_s"] == 25.0
    v = lambda h, q: {"prefix_cache_hits_total": h, "prefix_cache_queries_total": q, "generation_tokens_total": 1.0}
    rows = arena.sampler_rows([{"t": 0, "vllm": v(90, 100)}, {"t": 60, "vllm": v(186, 200)}], launched_epoch=0)
    assert "infra.prefix_hit" not in rows[0]["m"] and rows[1]["m"]["infra.prefix_hit"] == 0.96


def test_pending_resume_is_a_train_phase_until_step_one():
    """real-v6 27 Oct: a pending-only resume trains its saved groups first; its seq_start (before the replay) plays
    nothing, so no-play stays quiet until step 1 (N9's feed order)."""
    from arc_infra.runops import arena

    feed = [{"kind": "seq_resume", "epoch": 19.0, "step": 0, "pending_only": True, "pending_groups": 16},
            {"kind": "seq_start", "epoch": 113.0, "step": 0},
            {"kind": "seq_replay", "epoch": 118.0, "groups": 16, "resumed_pending": True},
            {"kind": "step", "epoch": 313.0, "step": 1},
            {"kind": "seq_phase", "epoch": 313.5, "step": 1, "stop": {}, "start": {"switch_s": 40.0}}]
    phases = [(r["t"], r["m"]["seq.phase"]) for r in arena.seq_feed_rows(feed) if "seq.phase" in r["m"]]
    assert phases == [(19.0, 0.0), (313.5, 1.0)]
    plain = [{"kind": "seq_resume", "epoch": 1.0, "step": 3, "pending_groups": 0}, {"kind": "seq_start", "epoch": 2.0}]
    assert [r["m"]["seq.phase"] for r in arena.seq_feed_rows(plain) if "seq.phase" in r["m"]] == [1.0]


def test_first_update_after_load_is_cold():
    """The feed's first one_call update follows the trainer load (real-v5 57.5 / 54.5 s, real-v6 59.3 / 28.7 s): it is
    logged as *_cold, outside the per-update targets; update 2 (warm) carries the judged keys."""
    from arc_infra.runops import arena

    feed = [{"kind": "seq_accumulate", "epoch": 10.0, "one_call": True, "call_util": 59.3, "setup_seconds": 28.7},
            {"kind": "seq_accumulate", "epoch": 20.0, "one_call": True, "call_util": 70.0, "setup_seconds": 4.0,
             "card_peak_gib": 247.0}]
    rows = [r["m"] for r in arena.seq_feed_rows(feed)]
    assert rows[0]["train.call_util_cold"] == 59.3 and "train.call_util" not in rows[0]
    assert rows[1]["train.call_util"] == 70.0 and rows[1]["train.setup_s"] == 4.0 and rows[1]["train.card_peak_gib"] == 247.0


V7_LAUNCH = 1793130372.0  # rl1-loop-seq-real-v7 (27 Oct), host_ram_gib from its cpulog, seq.phase from its feed
V7_SWITCH = {"total_gib": 1024, "switch": {"phase_s": 1800, "switch_gib": 50, "floor_gib": 120}}


def v7_host(alerts):
    return [(m, a) for m, a in alerts if a.rule == "host-switch"]


def test_real_v7_switch_projection_quiet_where_it_fit():
    """real-v7's first play phase: available host fell ~6 GiB/min to 261 GiB by the end; with the switch's 50 GiB it
    still cleared 120 (the read: the 960 anon+shm line was the wrong gauge): no host-switch alert."""
    r = rows("rl-real-v7-host.jsonl")
    out = ops_watch.backtest({"run": "v7", "launched": V7_LAUNCH, "targets": {}, "host": V7_SWITCH}, r,
                             until=max(x["t"] for x in r))
    assert v7_host(out) == []


def test_real_v7_switch_projection_fires_on_a_150_gib_smaller_host():
    """The same phase on a host 150 GiB smaller: the projection crosses the floor well before the switch, and not in
    the phase's first 5 min (too few points)."""
    r = rows("rl-real-v7-host.jsonl")
    plan = {"run": "v7", "launched": V7_LAUNCH, "targets": {}, "host": {**V7_SWITCH, "total_gib": 874}}
    hits = v7_host(ops_watch.backtest(plan, r, until=max(x["t"] for x in r)))
    play = next(x["t"] for x in r if (x.get("m") or {}).get("seq.phase") == 1)
    assert hits and (play - V7_LAUNCH) / 60 + 5 <= hits[0][0] < (max(x["t"] for x in r) - V7_LAUNCH) / 60 - 5
    assert "projected" in hits[0][1].text and "GiB/min" in hits[0][1].text


def test_heap_leg_api_stall_surfaces_the_guard_alert():
    """heap-a attempt 4 (27 Oct): tracemalloc's first snapshot pinned the API server, /metrics went unread 20 s+;
    the run's guard wrote a "guard blind" alert row at 22:44:26. The feed adapter passes it on: run-alert fires
    within one pass of it. A feed without alert rows (real-v7's host fixture) stays quiet on this rule."""
    from arc_infra.runops.arena import seq_feed_rows

    feed = rows("rl-heap-a4-feed.jsonl")
    r = seq_feed_rows(feed)
    blind = next(x for x in feed if x.get("kind") == "alert")["epoch"]
    launched = min(x["epoch"] for x in feed) - 60
    out = ops_watch.backtest({"run": "heap-a", "launched": launched, "targets": {}}, r, until=blind + 300)
    hits = [m for m, a in out if a.rule == "run-alert" and "guard blind" in a.text]
    assert hits and hits[0] <= (blind - launched) / 60 + 2
    quiet = ops_watch.backtest({"run": "v7", "launched": V7_LAUNCH, "targets": {}}, rows("rl-real-v7-host.jsonl"))
    assert "run-alert" not in rules(quiet)


def test_ahead_hit_share_fires_on_the_gate_legs_and_stays_quiet_when_taken():
    """gate-v1/v2 (28 Oct): the trainer took 0 of 192 early-encoded windows (encode.ahead_hits 0), a key mismatch
    in every rl_seq run, invisible for weeks. Their feeds carry an ahead write (seq_early_ahead / seq_ahead_pending)
    and a one-call update with no share: the each-rule on train.ahead_hit_share (min 0.9) fires on both; the same
    feed with the share at 0.97 stays quiet."""
    from arc_infra.runops.arena import seq_feed_rows

    plan = lambda launched: {"run": "g", "launched": launched, "targets": {}, "each": {"train.ahead_hit_share": {"min": 0.9}}}
    for name in ("rl-gate-v1-feed.jsonl", "rl-gate-v2-feed.jsonl"):
        feed = rows(name)
        launched = min(x["epoch"] for x in feed) - 60
        r = seq_feed_rows(feed)
        assert any("train.ahead_hit_share" in x["m"] for x in r)
        out = ops_watch.backtest(plan(launched), r, until=max(x["epoch"] for x in feed) + 120)
        assert [a for _, a in out if a.rule == "each:train.ahead_hit_share"], name
        good = [dict(x, ahead_hit_share=0.97, ahead_expected=True) if x.get("kind") == "seq_accumulate" else x
                for x in feed]
        quiet = ops_watch.backtest(plan(launched), seq_feed_rows(good), until=max(x["epoch"] for x in feed) + 120)
        assert not [a for _, a in quiet if a.rule == "each:train.ahead_hit_share"], name


def test_ahead_hit_share_reads_gate_v3_rows_as_hits_over_encoded():
    """gate-v3 (trunk a14120868) wrote ahead_misses = the trainer's encoded total (pre-encoded included): 12 hits of
    12 windows is 1.0, not the 0.5 its own share field said."""
    from arc_infra.runops.arena import seq_feed_rows

    feed = [{"kind": "seq_early_ahead", "epoch": 1.0, "specs": 24},
            {"kind": "seq_accumulate", "epoch": 2.0, "one_call": True, "ahead_expected": True, "ahead_hits": 12,
             "ahead_misses": 12, "ahead_hit_share": 0.5},
            {"kind": "seq_early_ahead", "epoch": 3.0, "specs": 24},
            {"kind": "seq_accumulate", "epoch": 4.0, "one_call": True, "ahead_expected": True, "ahead_hits": 12,
             "ahead_windows": 12, "ahead_hit_share": 1.0}]
    shares = [x["m"]["train.ahead_hit_share"] for x in seq_feed_rows(feed) if "train.ahead_hit_share" in x["m"]]
    assert shares == [1.0, 1.0]


def test_rl_arena_rows_read_the_exporters_mirror_without_fetching(tmp_path, monkeypatch):
    """An RL plan's sampler/cpulog/feed come from the mirror export_rl keeps fresh: no modal call of watch's own; the
    sampler's last-row epoch sets the time base (no volume mtime needed)."""
    from arc_infra.runops import volfetch

    monkeypatch.setattr(volfetch, "LOCKS", tmp_path)
    monkeypatch.setattr(ops_watch, "_run", lambda *a, **k: pytest.fail("watch fetched a fresh mirror itself"))
    monkeypatch.setattr(volfetch, "run", lambda *a, **k: pytest.fail("watch fetched a fresh mirror itself"))
    base = "validation/r/"
    rows = [{"t": 60, "epoch": 1000.0, "util": [90.0], "mem_gib": [200.0]},
            {"t": 120, "epoch": 1060.0, "util": [95.0], "mem_gib": [210.0]}]
    for name, body in (("live/sampler.jsonl", "\n".join(json.dumps(r) for r in rows)), ("util/cpulog.txt", ""),
                       ("arena/trials/rl-loop/rl/feed.jsonl", "")):
        p = volfetch.mirror("arc-prolong-runs", base + name)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(body)
    plan = {"run": "r", "launched": 1, "arena": {k: f"modal:arc-prolong-runs:{base}{v}" for k, v in (
        ("sampler", "live/sampler.jsonl"), ("cpulog", "util/cpulog.txt"), ("feed", "arena/trials/rl-loop/rl/feed.jsonl"))}}
    got = ops_watch.arena_rows(plan)
    assert [r["t"] for r in got] == [1000.0, 1060.0]


def test_each_sustained_needs_the_rate_to_confirm():
    """real-v11 28 Oct: api_cpu 1.02 on one server while gen ran 2,900-3,300 tok/s (plan 2,900) fired "each" and woke
    the arm for nothing. With sustain_s + confirm it fires only when the API stays past one core for 10 min AND gen
    tok/s is under 0.8 x plan."""
    L = 1000.0
    plan = {"run": "a", "launched": L, "boot_s": 60, "cadence_s": 60, "targets": {"infra.gen_tok_s": 2900},
            "each": {"infra.api_cpu": {"max": 1.0, "sustain_s": 600, "confirm": {"infra.gen_tok_s": 0.8}}}}

    def rows(cpu, gen):
        return [{"t": L + 60 * i, "m": {"infra.gen_tok_s": gen(i), "infra.gen_tokens": 3000.0 * i,
                                        "infra.api_cpu": cpu(i)}} for i in range(1, 40)]

    def fired(r, now=L + 60 * 39):
        return [a for _, a in ops_watch.backtest(plan, r, until=now) if a.rule == "each:infra.api_cpu"]

    assert not fired(rows(lambda i: 1.02 + 0.01 * (i % 3), lambda i: 3000.0))  # v11: pinned API, gen at plan
    assert fired(rows(lambda i: 1.05, lambda i: 3000.0 if i < 20 else 1800.0))  # pinned and gen 62% of plan
    assert not fired(rows(lambda i: 0.9 if i % 7 == 0 else 1.05, lambda i: 1800.0))  # not sustained: under 1 within
    plain = {**plan, "each": {"infra.api_cpu": {"max": 1.0}}}  # the old per-value form still fires on one value
    assert [a for _, a in ops_watch.backtest(plain, rows(lambda i: 1.02, lambda i: 3000.0), until=L + 600)
            if a.rule == "each:infra.api_cpu"]


def test_confirm_dict_flags_low_running_only_when_kv_is_free_and_requests_wait():
    """real-v11 28 Oct: running 237 vs plan 304 at KV 0.79 with gen on plan (the KV gate binding as contexts grow)
    fired plan@20m. With confirm {"infra.running": {"infra.kv_frac": {"max": 0.7}, "infra.waiting": {"min": 0}}} low
    running is flagged only while KV is under 0.7 and requests wait."""
    L = 1000.0
    plan = {"run": "k", "launched": L, "boot_s": 60, "cadence_s": 60, "rate_floor": 0.8,
            "targets": {"infra.running": 304, "infra.gen_tok_s": 2900},
            "confirm": {"infra.running": {"infra.kv_frac": {"max": 0.7}, "infra.waiting": {"min": 0}}}}

    def rows(kv, waiting):
        return [{"t": L + 60 * i, "m": {"infra.running": 237.0, "infra.gen_tok_s": 2900.0, "infra.gen_tokens": 2900.0 * i,
                                        "infra.kv_frac": kv, "infra.waiting": waiting}} for i in range(1, 40)]

    def marks(r):
        return [a.rule for _, a in ops_watch.backtest(plan, r, until=L + 60 * 39) if a.rule.startswith(("plan@", "rate:"))]

    assert not marks(rows(0.79, 0.0))  # v11: KV-gated
    assert not marks(rows(0.79, 3.0))  # KV full, a queue: still the gate
    assert not marks(rows(0.5, 0.0))  # KV free, nothing waiting: too few tries offered, not a serving fault
    assert marks(rows(0.5, 3.0))  # KV free and requests waiting: the engine is not admitting


def test_plan_marks_respect_phase_from_s():
    """real-v12 (28 Oct): plan@10m read kv_frac over min 5-10 of a lockstep phase start (0.48 ramping); a target with
    phase_from_s is judged by the marks only from that far into the phase (plan@20m reads min 15-20)."""
    L = 1000.0
    plan = {"run": "k", "launched": L, "boot_s": 60, "cadence_s": 60, "rate_floor": 0.8,
            "targets": {"infra.running": 100, "infra.kv_frac": {"plan": 0.8, "steady": True, "phase_from_s": 600}}}
    rows = lambda kv20: [{"t": L + 60 * i, "m": {"infra.running": 100.0, "infra.gen_tok_s": 50.0,
                                                 "infra.gen_tokens": 50.0 * i,
                                                 "infra.kv_frac": 0.4 if i < 15 else kv20}} for i in range(1, 30)]
    def fired(kv20, now):
        w = ops_watch.Watch(plan)
        return [a.rule for a in w.check(Snapshot(now=L + now, rows=rows(kv20), live=True))
                if "kv_frac" in a.text]
    assert "plan@10m" not in fired(0.75, 60 * 12)   # ramp before phase_from_s: not judged
    assert "plan@20m" not in fired(0.75, 60 * 22)   # min 15-20 at 0.75 >= 0.64
    assert "plan@20m" in fired(0.4, 60 * 22)        # still low past phase_from_s: fires


def test_arc2_fork_replies_per_attempt_read_once_at_10_min():
    """arc2 29 Oct: a stale fork bundle answers once per attempt (fk-naive2 p2: 77/77 at 554 s, 87/87 at 674 s); a
    healthy one reads 3.4-4.1 at the first snapshot past 10 min of play but 2.5 at 553 s (fk-naive3 p3)."""
    from arc_infra.runops import arena

    snap = lambda el, rep, att: {"elapsed_seconds": el, "tokens_per_second": 1000, "replies": rep,
                                 "where_tokens_go": {"by_attempt": {"1": {"attempts": att}}}}
    plan = {"run": "fk", "launched": 0, "boot_s": 600, "cadence_s": 120, "targets": {"infra.gen_tok_s": 1000},
            "each": {"play.replies_per_attempt": {"min": 3, "after_s": 600, "once": True}}}
    feeds = {"stale": [(70, 0, 0), (433, 20, 20), (554, 77, 77), (674, 87, 87), (795, 95, 95)],
             "healthy": [(70, 0, 0), (432, 75, 25), (553, 181, 73), (673, 301, 74), (794, 373, 74)]}
    assert arena.arc2_row(snap(0, 0, 0), 0)["m"].get("play.replies_per_attempt") is None
    fired = {}
    for name, feed in feeds.items():
        rows = [arena.arc2_row(snap(*x), 1000 + x[0]) for x in feed]
        w = Watch(plan)
        fired[name] = [(r["t"] - 1000, a.rule) for i, r in enumerate(rows)
                       for a in w.check(Snapshot(now=r["t"], rows=rows[:i + 1], live=True))
                       if a.rule.startswith("each:")]
    assert fired == {"stale": [(674, "each:play.replies_per_attempt")], "healthy": []}


def test_served_share_and_encode_errors_each_rules():
    """v17 a3 (30 Oct): U19's encoders raised 5 FileNotFoundErrors (re-parsed silently) and windows trained without
    served routes, unreported. The rules fire on low coverage, errors, or a call with no coverage reading; quiet on
    full coverage and on rows from before the fields existed."""
    from arc_infra.runops.arena import seq_feed_rows

    def feed(**row):
        return [{"kind": "seq_start", "epoch": 1000.0},
                {"kind": "seq_accumulate", "one_call": True, "epoch": 2000.0, **row}]

    plan = {"run": "g", "launched": 900.0, "targets": {},
            "each": {"train.served_share": {"min": 0.9}, "train.encode_errors": {"max": 0}}}
    cases = {"low": ({"encode_errors": 0, "served_share": 0.4}, "train.served_share"),
             "errors": ({"encode_errors": 5, "served_share": 1.0}, "train.encode_errors"),
             "unrouted": ({"encode_errors": 0}, "train.served_share")}
    for name, (row, key) in cases.items():
        out = ops_watch.backtest(plan, seq_feed_rows(feed(**row)), until=2200.0)
        assert [a for _, a in out if a.rule == f"each:{key}"], name
    for row in ({"encode_errors": 0, "served_share": 1.0}, {"ahead_hit_share": 1.0}):
        out = ops_watch.backtest(plan, seq_feed_rows(feed(**row)), until=2200.0)
        assert not [a for _, a in out if a.rule.startswith("each:")], row


def test_boot_needs_a_second_signal_and_a_failed_read_after_play_stays_quiet():
    """arc3sub 30 Oct: a failed read gave rows=[] and fired [boot] 1,000 min into healthy runs, on all 3 watchers."""
    from arc_infra.runops import arena

    r = arena.progress_rows(rows("psfgab-v2-progress.jsonl"))
    launched = r[0]["t"] - 60
    plan = {"run": "psfgab-v2", "launched": launched, "boot_s": 300, "cadence_s": 60,
            "targets": {"infra.running": {"plan": 40, "steady": True}, "infra.gen_tok_s": {"plan": 1500, "steady": True}}}
    mid = r[len(r) // 2]["t"]
    # healthy, two blips with no kept rows (a restarted watcher): no boot, whatever the blip's length
    blips = [(mid, mid + 120), (mid + 600, mid + 3000)]
    assert "boot" not in rules(ops_watch.backtest(plan, r, read_fails=blips))
    # state from before the fix: marks done means play was seen
    w = Watch(plan)
    w.restore({"marks_done": [10, 20]})
    assert w.check(Snapshot(now=launched + 90_000, rows=[], read_failed=True)) == []
    assert w.check(Snapshot(now=launched + 91_000, rows=[])) == []
    # past the deadline, before play: quiet until reads have failed READ_FAIL_S
    w = Watch({"run": "x", "launched": 0, "boot_s": 600, "targets": {"infra.running": 10}})
    assert w.check(Snapshot(now=1000, rows=[], read_failed=True)) == []
    assert w.check(Snapshot(now=1800, rows=[], read_failed=True)) == []
    assert [a.rule for a in w.check(Snapshot(now=1900, rows=[], read_failed=True))] == ["boot"]
    # the same blip, then a good read with play: boot is over for good
    w = Watch({"run": "x", "launched": 0, "boot_s": 600, "targets": {"infra.running": 10}})
    assert w.check(Snapshot(now=1000, rows=[], read_failed=True)) == []
    assert w.check(Snapshot(now=1100, rows=[{"t": 1090, "m": {"infra.running": 5}}])) == []
    assert w.check(Snapshot(now=9000, rows=[], read_failed=True)) == []
    assert w.state()["played"] is True and w.state()["read_fail_since"] == 9000


def test_a_bad_boot_still_fires_on_time():
    bad = (FIX / "psfg-bootstrap.log").read_text()
    plan = {"run": "psfg-v1", "launched": 0, "boot_s": 900, "cadence_s": 60, "targets": {"infra.running": 40}}
    # the file was never written: every read fails from launch
    out = ops_watch.backtest(plan, [], log_text=bad, until=1500, live_until=10**9, read_fails=[(0, 1500)])
    assert min(m for m, a in out if a.rule == "boot") == 22.5 + 0.5  # the first pass past 1.5 x boot
    # a read that works and shows the server at 0: fires at once
    zero = [{"t": float(t), "m": {"infra.running": 0.0}} for t in range(0, 1500, 60)]
    out = ops_watch.backtest(plan, zero, until=1500)
    assert min(m for m, a in out if a.rule == "boot") == 23.0


def test_a_failed_read_keeps_the_last_good_rows(monkeypatch):
    prog = rows("psfgab-v2-progress.jsonl")
    text = "\n".join(json.dumps(x) for x in prog)
    spec = "nebius:validation/kept/arena/live/progress.jsonl"
    plan = {"run": "kept", "arena": {"progress": spec}}
    got = {"v": text}
    monkeypatch.setattr(ops_watch, "fetch", lambda s, **k: (got["v"], 5.0))
    monkeypatch.setattr(ops_watch, "KEPT", {})
    ops_watch.READ_FAILED.clear()
    good = ops_watch.arena_rows(plan)
    assert good and not ops_watch.READ_FAILED
    got["v"] = None
    assert ops_watch.arena_rows(plan) == good and ops_watch.READ_FAILED == {spec}
    ops_watch.READ_FAILED.clear()
    got["v"] = ""  # read fine, empty: not a failure, and rows do not vanish
    assert ops_watch.arena_rows(plan) == good and not ops_watch.READ_FAILED


def test_sft_warmup_keep_layers_down_is_quiet_until_the_first_step():
    def crash(text):
        return [a for a in Watch({"run": "s", "launched": 0}).check(Snapshot(now=60, rows=[], live=True, log_text=text))
                if a.rule == "crash"]
    for name in ("sft-c731-boot.jsonl", "sft-joint-boot.jsonl"):  # designed boot step-downs (4 -> 1 / 2), then steps
        assert crash((FIX / name).read_text()) == []
    late = (FIX / "sft-c731-boot.jsonl").read_text() + json.dumps({"event": "keep_layers_down", "keep_layers": 0,
                                                                   "why": "CUDA out of memory. Tried to allocate 1 GiB"}) + "\n"
    assert len(crash(late)) == 1  # after a train step it is an alert again
    assert any("OutOfMemoryError" in a.text for a in crash((FIX / "sft-score-oom.log").read_text()))  # a step OOM


def test_sft_read_pause_is_quiet_with_progress_pauses_and_a_stall_still_fires():
    """sft-p50d7-a2r2t-c942-v3 2 Nov (healthy): the slice + ARC-2 reads after step 960 held the step 160 s against a
    4 x 37.6 s limit and fired "stuck at 960". progress_pauses (every 24: slice, every 32: ARC-2) lift the limit at those
    steps only: a hang after a non-read step, or a read step past limit + pause, still fires."""
    r = rows("sft-a2r2t-v3-read.jsonl")
    plan = {"run": "v3", "launched": 1793622543, "boot_s": 1200, "cadence_s": 15, "silence_s": 900,
            "progress_key": "train.step", "progress_k": 4}
    pauses = {**plan, "progress_pauses": [{"every": 24, "s": 180}, {"every": 32, "s": 300}]}
    end = r[-1]["t"]
    assert "progress:train.step" in rules(ops_watch.backtest(plan, r, live_until=10**10, until=end, step=15))
    assert "progress:train.step" not in rules(ops_watch.backtest(pauses, r, live_until=10**10, until=end, step=15))

    def hang(upto, minutes):  # the feed's infra rows go on (app up, cards busy), the step stops
        kept = [x for x in r if x["t"] <= upto]
        return kept + [{"t": upto + 15 * i, "m": {"infra.gpu_util.0": 97.0}} for i in range(1, minutes * 4)]
    at961 = hang(1793624330, 6)  # after step 961 (no read due)
    assert "progress:train.step" in rules(ops_watch.backtest(pauses, at961, live_until=10**10, until=at961[-1]["t"], step=15))
    at960 = hang(1793624227, 15)  # inside the step-960 read, 15 min > 2.5 + 8 min
    assert "progress:train.step" in rules(ops_watch.backtest(pauses, at960, live_until=10**10, until=at960[-1]["t"], step=15))
