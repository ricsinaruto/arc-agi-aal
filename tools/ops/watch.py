"""Shared run watcher. Run it as a background task of the owning agent from launch: every
line it prints wakes that agent. It never stops a run. Rules: src/arc_infra/runops/watch.py.

  watch.py run PLAN.json [--interval 120]   live watch until the run ends (exit 0 after "END")
  watch.py stop RUN                          ask that watcher to exit (it ignores SIGTERM/SIGHUP: a broad `pkill -f`
                                             once killed every arm's watcher; stop one by pid with `kill -9` only)
  watch.py beats                             watchers whose heartbeat is stale (dash loop, each pass; exit 1 if any)
  watch.py backtest PLAN.json --metrics F [--log F] [--live-until T] [--beat-until T] [--remote-until T] [--until T]

Remote specs in the plan: "modal:<volume>:<path>", "nebius:<path under runs/>" or "modal-app:<app id>" (its logs).
A run whose app/VM goes without its own clean end (restart-guard status ok, a plan "end_marker" / default END_MARKER
line in its logs or rows, a Nebius final.json that is not failed) is an ALERT [unclean-end] and the watcher exits 3,
never a plain END. Plan keys used here beyond the rules:
  platform (modal|nebius), app_id (Modal), vm (Nebius VM name; default arc-<run>), metrics_remote (the run's own
  train-metrics jsonl, mirrored into <scratch>/train-metrics/<run>.jsonl; leave unset when a feed writes that file),
  runops_state (the restart guard's attempts.json), logs (list of remote log specs scanned for crash signatures).
"""
import argparse
import datetime
import json
import os
import pathlib
import re
import signal
import subprocess
import sys
import time

sys.path.insert(0, str(pathlib.Path(__file__).parent))
import paths  # noqa: E402

sys.path.insert(0, str(paths.ROOT_PATH / "src"))
from arc_infra.runops import arena  # noqa: E402
from arc_infra.runops._io import read_json, write_json  # noqa: E402
from arc_infra.runops.watch import END_MARKER, Snapshot, Watch, validate_plan  # noqa: E402

MODAL = str(paths.ROOT_PATH / ".venv/bin/modal")
BUCKET = "nebius-arc:arc-runs-bucket/runs/"
TAIL = 400_000


def _run(cmd, timeout=120):
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
        return r.stdout if r.returncode == 0 else None
    except (subprocess.SubprocessError, OSError):
        return None


def fetch(spec, tail=TAIL):
    """(text or None, mtime epoch or None) of a remote file."""
    kind, _, rest = spec.partition(":")
    if kind == "modal-app":  # `modal app logs` follows a live app: take what arrives in 20 s
        try:
            r = subprocess.run([MODAL, "app", "logs", rest], capture_output=True, text=True, timeout=20)
            text = r.stdout + r.stderr
        except subprocess.TimeoutExpired as e:
            text = (e.stdout or b"").decode(errors="replace") if isinstance(e.stdout, bytes) else (e.stdout or "")
        except OSError:
            text = None
        return (text or "")[-tail:] or None, None
    if kind == "nebius":
        meta = _run(["rclone", "lsjson", BUCKET + rest])
        text = _run(["rclone", "cat", "--tail", str(tail), BUCKET + rest])
        try:
            mt = datetime.datetime.fromisoformat(json.loads(meta)[0]["ModTime"]).timestamp()
        except (TypeError, ValueError, IndexError, KeyError):
            mt = None
        return text, mt
    if kind == "modal":
        vol, _, path = rest.partition(":")
        meta = _run([MODAL, "volume", "ls", "--json", vol, path])
        local = paths.SCRATCH / "watch" / "fetch" / vol / path
        local.parent.mkdir(parents=True, exist_ok=True)
        ok = _run([MODAL, "volume", "get", "--force", vol, path, str(local)], timeout=300) is not None
        text = local.read_text(errors="replace")[-tail:] if ok and local.is_file() else None
        try:  # "2026-10-24 10:59 CEST": minute resolution, local zone
            stamp = json.loads(meta)[0]["created_modified"].rsplit(" ", 1)[0]
            mt = datetime.datetime.strptime(stamp, "%Y-%m-%d %H:%M").astimezone().timestamp()
        except (TypeError, ValueError, IndexError, KeyError):
            mt = None
        return text, mt
    raise ValueError(f"remote spec {spec!r}: modal:<volume>:<path> or nebius:<path>")


def liveness(plan):
    if plan.get("platform") == "modal" and plan.get("app_id"):
        out = _run([MODAL, "app", "list", "--json"], timeout=60)
        if out is None:
            return None
        apps = [{k.lower().replace(" ", "_"): v for k, v in a.items()} for a in json.loads(out)]
        return any(a.get("app_id") == plan["app_id"] and a.get("state") != "stopped" for a in apps)
    if plan.get("platform") == "nebius":
        import reconcile

        vms = reconcile.nebius_vms()
        if vms is None:
            return None
        name = (plan.get("vm") or plan["run"]).lower().removeprefix("arc-")[:59]
        return any(k.startswith(name) for k in vms)
    return None


def recorded_ended(plan):
    q = paths.read_json(paths.ROOT_PATH / "docs/modal-queue.json")
    return plan["run"] not in {e.get("run") for e in q.get("live", []) + q.get("queued", [])}


def read_rows(path):
    rows = []
    try:
        for line in pathlib.Path(path).read_text().splitlines():
            try:
                rows.append(json.loads(line))
            except ValueError:
                continue
    except FileNotFoundError:
        pass
    return rows


def snapshot(plan, metrics_path):
    now = time.time()
    remote_age = beat_age = guard_status = None
    incidents = []
    if plan.get("metrics_remote"):
        text, mt = fetch(plan["metrics_remote"], tail=50_000_000)
        if text:
            metrics_path.write_text(text)
        remote_age = None if mt is None else now - mt
    if plan.get("runops_state"):
        text, _ = fetch(plan["runops_state"])
        try:
            st = json.loads(text or "")
            incidents = st.get("incidents") or []
            last = (st.get("attempts") or [{}])[-1]
            guard_status = (last.get("ended") or {}).get("status")
            live_attempts = [a for a in st.get("attempts") or [] if a.get("ended") is None]
            if live_attempts:
                beat_age = now - float(live_attempts[-1]["last_beat"])
        except (ValueError, KeyError, TypeError):
            pass
    specs = list(plan.get("logs") or [])
    if plan.get("platform") == "modal" and plan.get("app_id") and not any(x.startswith("modal-app:") for x in specs):
        specs.append(f"modal-app:{plan['app_id']}")
    live = liveness(plan)
    logs = "\n".join(fetch(spec)[0] or "" for spec in specs)
    rows = read_rows(metrics_path) + arena_rows(plan)
    clean = clean_end(plan, rows, logs, guard_status) if live is False else None
    return Snapshot(now=now, rows=rows, live=live, remote_age=remote_age, beat_age=beat_age, incidents=incidents,
                    log_text=logs, ended=recorded_ended(plan) and live is not True, clean_end=clean)


def clean_end(plan, rows, logs, guard):
    """True when the run left its own clean-end evidence."""
    marker = re.compile(END_MARKER + (f"|{plan['end_marker']}" if plan.get("end_marker") else ""))
    if guard == "ok" or marker.search(logs or ""):
        return True
    if any(marker.search(str(r.get("note") or "")) or marker.search(str(r.get("alert") or "")) for r in rows):
        return True
    if plan.get("platform") == "nebius":
        text, _ = fetch(f"nebius:validation/{plan['run']}/nebius/final.json")
        try:
            return bool(text) and "fail" not in str(json.loads(text).get("state", json.loads(text).get("status", ""))).lower()
        except ValueError:
            return False
    return False


def arena_rows(plan):
    """Plan "arena": {"progress": spec of arena/live/progress.jsonl} (ARC-AGI-3 arena) or {"arc2": spec of
    arena/live/arc2-<trial>.json} (ARC-AGI-2 roster; each new snapshot is kept in <scratch>/watch/<run>.arena.jsonl)."""
    a = plan.get("arena") or {}
    if a.get("progress"):
        text, _ = fetch(a["progress"], tail=20_000_000)
        return arena.progress_rows([json.loads(x) for x in (text or "").splitlines() if x.strip().startswith("{")])
    if a.get("arc2"):
        hist = paths.SCRATCH / "watch" / f"{plan['run']}.arena.jsonl"
        hist.parent.mkdir(parents=True, exist_ok=True)
        snaps = read_rows(hist)
        text, mt = fetch(a["arc2"])
        if text and mt and (not snaps or mt > snaps[-1]["t"]):
            try:
                snaps.append({"t": mt, "snap": json.loads(text)})
                with open(hist, "a") as f:
                    f.write(json.dumps(snaps[-1]) + "\n")
            except ValueError:
                pass
        out, prev = [], None
        for x in snaps:
            out.append(arena.arc2_row(x["snap"], x["t"], prev))
            prev = (x["t"], x["snap"].get("completion_tokens"))
        return out
    return []


def emit(plan, alerts, metrics_path, now):
    log = paths.SCRATCH / "watch" / f"{plan['run']}.log"
    log.parent.mkdir(parents=True, exist_ok=True)
    stamp = time.strftime("%H:%M", time.gmtime(now))
    for a in alerts:
        line = f"ALERT {plan['run']} [{a.rule}] {a.text}"
        print(f"{stamp} {line}", flush=True)
        with open(log, "a") as f:
            f.write(f"{stamp} {line}\n")
        with open(metrics_path, "a") as f:  # shown on training-live.html
            f.write(json.dumps({"t": now, "m": {}, "alert": f"watch: [{a.rule}] {a.text}"}) + "\n")


WDIR = paths.SCRATCH / "watch"


def _files(run):
    return {k: WDIR / f"{run}.{k}" for k in ("beat.json", "state.json", "stop")}


def _ignore(signum, _frame):
    print(f"watch: ignored {signal.Signals(signum).name} (a broad kill?); stop with: watch.py stop RUN", flush=True)


def follow(plan, rec):
    """(plan for this pass, attempt): a follow_record plan takes launched/app_id from the record's current attempt."""
    if not plan.get("follow_record") or not rec or not rec.get("ledger"):
        return plan, None
    row = rec["ledger"][-1]["row"]
    eff = {**plan, "launched": row.get("start_epoch") or plan.get("launched")}
    if str(row.get("app_id") or "").startswith("ap-"):
        eff["app_id"] = row["app_id"]
    return eff, int(rec.get("attempt") or 1)


def cmd_run(a):
    """Survives what killed the first watchers: SIGTERM/SIGHUP from broad `pkill -f` patterns are ignored and a failing
    pass is reported and retried. Rule state persists, so a restarted watcher neither re-fires nor forgets its marks;
    a heartbeat file lets the dash loop (``beats``) flag a watcher that died anyway (SIGKILL, reboot)."""
    from arc_infra.runops import records

    base = validate_plan(json.loads(pathlib.Path(a.plan).read_text()))
    plan, attempt = follow(base, records.get(base["run"]))
    w = Watch(plan)
    f = _files(plan["run"])
    WDIR.mkdir(parents=True, exist_ok=True)
    f["stop"].unlink(missing_ok=True)
    try:
        st = json.loads(f["state.json"].read_text())
        if st.get("attempt") == attempt:  # a new attempt starts with fresh rule state
            w.restore(st)
    except (OSError, ValueError):
        pass
    for sig in (signal.SIGTERM, signal.SIGHUP):
        signal.signal(sig, _ignore)
    metrics = paths.SCRATCH / "train-metrics" / f"{plan['run']}.jsonl"
    metrics.parent.mkdir(parents=True, exist_ok=True)
    print(f"watching {plan['run']} every {a.interval}s (silence {w.silence_s:.0f}s, boot {w.boot_s:.0f}s, "
          f"freeze {w.freeze_s:.0f}s)", flush=True)
    last_err, ended = None, False
    while not f["stop"].exists():
        try:
            nplan, natt = follow(base, records.get(base["run"])) if base.get("follow_record") else (plan, attempt)
            if natt != attempt:
                print(f"watch: {plan['run']} attempt {natt} (was {attempt}): rule state reset", flush=True)
                w = Watch(nplan)
            plan, attempt = nplan, natt
            s = snapshot(plan, metrics)
            if plan.get("follow_record") and plan.get("launched"):  # an earlier attempt's rows are not this one's
                s.rows = [r for r in s.rows if (r.get("t") or 0) >= float(plan["launched"]) - 60]
            emit(plan, w.check(s), metrics, s.now)
            write_json(f["state.json"], {**w.state(), "attempt": attempt})
            ended, last_err = s.ended or (s.live is False and s.clean_end is False and "unclean-end" in
                                          {k[0] for k in w.seen}), None
        except Exception as e:  # a bad fetch or parse must not end the watch
            if repr(e) != last_err:
                print(f"watch: pass failed, retrying: {e!r}", flush=True)
            last_err = repr(e)
        write_json(f["beat.json"], {"run": plan["run"], "pid": os.getpid(), "t": time.time(),
                                          "interval": a.interval, "plan": str(pathlib.Path(a.plan).resolve()),
                                          "arm": plan.get("arm"), "ended": ended})
        if ended:
            if s.clean_end:
                print(f"END {plan['run']}: clean end, no live record, app/VM down", flush=True)
                return 0
            print(f"STOPPED WATCHING {plan['run']}: app/VM gone WITHOUT a clean end (see the unclean-end alert)",
                  flush=True)
            return 3
        deadline = time.time() + a.interval
        while time.time() < deadline and not f["stop"].exists():
            time.sleep(min(5.0, a.interval))
    print(f"STOPPED {plan['run']} (watch.py stop)", flush=True)
    write_json(f["beat.json"], {**read_json(f["beat.json"], {}), "ended": True, "stopped": True})
    return 0


def cmd_stop(a):
    WDIR.mkdir(parents=True, exist_ok=True)
    _files(a.run)["stop"].touch()
    print(f"stop requested for {a.run}'s watcher (exits within 5 s)")


def stale_beats(now=None, wdir=WDIR, live_runs=None):
    """[(run, arm, age_s, pid alive)] for watchers whose heartbeat is older than max(3 x interval, 600 s) while their run
    is still live (a queue row). Beat files of ended runs (record settled, or the watcher exited: END, unclean-end,
    stop) are deleted."""
    now, out = now or time.time(), []
    if live_runs is None:
        q = paths.read_json(paths.ROOT_PATH / "docs/modal-queue.json")
        live_runs = {e.get("run") for e in q.get("live", []) + q.get("queued", [])}
    for p in sorted(wdir.glob("*.beat.json")):
        try:
            b = json.loads(p.read_text())
        except (OSError, ValueError):
            continue
        if b.get("ended") or b.get("run") not in live_runs:
            p.unlink(missing_ok=True)
            continue
        age = now - float(b.get("t", 0))
        if age <= max(3 * float(b.get("interval", 120)), 600):
            continue
        try:
            os.kill(int(b["pid"]), 0)
            alive = True
        except (OSError, KeyError, ValueError):
            alive = False
        out.append((b.get("run"), b.get("arm"), age, alive))
    return out


def cmd_beats(_a):
    bad = stale_beats()
    for run, arm, age, alive in bad:
        print(f"STALE WATCHER {run} ({arm}): no heartbeat for {age / 60:.0f} min, process "
              f"{'hung' if alive else 'gone'}; the arm restarts it: watch.py run <plan> (state is kept)")
    return 1 if bad else 0


def backtest(plan, rows, log_text="", live_until=None, beat_until=None, remote_until=None, until=None, step=60.0,
             log_at=None, clean=None):
    """Replay: [(minutes since launch, Alert)]. *_until: when that signal stopped (None: it never did)."""
    w = Watch(plan)
    t0 = float(plan["launched"] if plan.get("launched") is not None else min(r["t"] for r in rows))
    end = until or max([r["t"] for r in rows] or [t0]) + 2 * w.silence_s
    out, now = [], t0
    while now <= end:
        s = Snapshot(now=now, rows=rows, live=None if live_until is None else now <= live_until,
                     remote_age=None if remote_until is None else max(0.0, now - min(now, remote_until)),
                     beat_age=None if beat_until is None else max(0.0, now - min(now, beat_until)),
                     log_text=log_text if log_at is None or now >= log_at else "",
                     clean_end=None if live_until is None or now <= live_until else
                     (clean if clean is not None else bool(re.search(END_MARKER, log_text or ""))))
        out += [(round((now - t0) / 60, 1), x) for x in w.check(s)]
        now += step
    return out


def cmd_backtest(a):
    plan = validate_plan(json.loads(pathlib.Path(a.plan).read_text()))
    rows = read_rows(a.metrics)
    text = pathlib.Path(a.log).read_text(errors="replace") if a.log else ""
    for minute, x in backtest(plan, rows, text, a.live_until, a.beat_until, a.remote_until, a.until):
        print(f"+{minute:6.1f} min [{x.rule}] {x.text}")


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("plan")
    r.add_argument("--interval", type=float, default=120)
    st = sub.add_parser("stop")
    st.add_argument("run")
    sub.add_parser("beats")
    b = sub.add_parser("backtest")
    b.add_argument("plan")
    b.add_argument("--metrics", required=True)
    b.add_argument("--log")
    for k in ("live-until", "beat-until", "remote-until", "until"):
        b.add_argument(f"--{k}", type=float)
    a = ap.parse_args()
    return {"run": cmd_run, "stop": cmd_stop, "beats": cmd_beats, "backtest": cmd_backtest}[a.cmd](a)


if __name__ == "__main__":
    sys.exit(main())
