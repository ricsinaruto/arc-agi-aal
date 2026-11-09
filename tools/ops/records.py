"""Run records CLI (docs/runs/<run>.json; library: src/arc_infra/runops/records.py). Always acts on the main checkout,
also when called from an arm worktree.

  records.py reserve RUN --arm A --platform modal|nebius --gpu "2 x B300" [--gpu-count 2] --bound USD --max-seconds S
                     --auth TEXT --what TEXT [--kind K] [--long-run-24h] [--set key=json ...]   gate + ledger + live row
  records.py settle RUN STATE --cause CAUSE [--set key=json ...]   end the current attempt: STATE is the ledger's
                     billing state (completed, stopped, failed); CAUSE why it ended (required): completed |
                     stopped-owner | preempt | capacity | superseded | bug:<class>. Again on an ended run: corrects it
  records.py require RUN                                 exit 0 only if RUN holds a fresh open reservation (launchers:
                                                         right before the cloud call; exit 1 = do not launch)
  records.py running RUN [--set app_id=ap-...]           the run is up (an app_id ap-... via `ledger` does the same)
  records.py unclassified                                ended runs with no stated cause (exit 1 if any)
  A relaunch of an ended run: `reserve` the same RUN again -> attempt N+1 in the same record (<run>/attempt-N).
  records.py queue RUN --set what="..." [--set next="..."]   edit the run's queue row
  records.py ledger RUN --set app_id=ap-... [--set k=json ...]   patch the run's open ledger row
  records.py cpu-app APP_ID                              register a CPU-only Modal app
  records.py cpu-app-drop APP_ID                         remove a CPU-app-only registration (mistyped id)
  records.py cpu-autoreg                                 register live CPU-only Modal apps that have no row
  records.py gate --arm A --platform P --gpu G [--gpu-count N] --bound USD    dry check, exit 1 on refusal
  records.py result RUN --metric score300k=13.7 [--metric tok_s=1534] [--bundle ID] [--vs RUN] [--verdict "..."]
                                                         one read of the run, kept in its record (at each read)
  records.py results [--arm A] [--since 6h|2026-10-24|EPOCH] [--metric M] [--prefix P] [--json]   query the reads
  records.py attach RUN LOG                              after `modal run --detach ... > LOG`: the app id in LOG goes
                                                         into the record (queue + ledger); exit 1 loudly without one
  records.py out RUN                                     the current attempt's output folder name (<run> or
                                                         <run>-attempt<N>): knobs, exporters and watchers use it
  records.py show RUN | sync | build | billing (Modal's bill to date -> docs/budget.json, for the gate)
  records.py snapshot                                    ended records -> docs/runs-settled/ (local); budget.json commit, pull, push
  records.py pull                                        `git pull` + push of the main checkout
  records.py push RUN...                                 no-op (records are untracked since 24 Oct)
Records and the views (docs/spend.json, docs/modal-queue.json) are machine-local and untracked; edit only through here.
"""
import argparse
import json
import re
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from arc_infra.runops import records as R  # noqa: E402


def kv(items):
    out = {}
    for item in items or []:
        k, eq, v = item.partition("=")
        if not eq or not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_.-]*", k):
            raise SystemExit(f"--set {item[:80]!r}: key=value (value as JSON or text), one field per --set")
        try:
            out[k] = json.loads(v)
        except ValueError:
            out[k] = v
    return out


def push(runs):
    print("records are machine-local since 24 Oct (untracked): nothing to push")


def _git(*args, check=False):
    return subprocess.run(["git", "-C", str(R.ROOT), *args], capture_output=True, text=True, check=check)


def pull():
    """`git pull` of the main checkout (records and views are untracked, so they never conflict), then push."""
    for _ in range(3):
        r = _git("pull", "--no-rebase", "-q", "--no-edit", "origin", "main")
        if r.returncode:
            bad = _git("diff", "--name-only", "--diff-filter=U").stdout.split()
            _git("merge", "--abort")
            sys.exit(f"PULL FAILED (merge aborted): {' '.join(bad) or r.stderr.strip()[-300:]}")
        if _git("push", "-q", "origin", "HEAD:main").returncode == 0:
            return print(f"pulled and pushed ({_git('log', '--oneline', '-1').stdout.strip()[:60]})")
    sys.exit("PUSH FAILED after 3 pulls (committed locally)")


def snapshot():
    """Daily: ended records -> docs/runs-settled/ (local, untracked); docs/budget.json committed and pushed."""
    n = R.snapshot()
    files = ["docs/budget.json"]
    _git("add", "--", *files, check=True)
    if _git("diff", "--cached", "--quiet", "--", *files).returncode == 0:
        return print(f"snapshot: nothing new ({n} written)")
    _git("commit", "-q", "-m", f"records: settled snapshot ({n} changed)", "--", *files, check=True)
    pull()


def since_epoch(text):
    """'6h' / '2d' / '90m' ago, an ISO date or time (UTC), or an epoch."""
    import datetime
    import re
    import time

    if not text:
        return 0.0
    m = re.fullmatch(r"(\d+(?:\.\d+)?)([mhd])", text)
    if m:
        return time.time() - float(m.group(1)) * {"m": 60, "h": 3600, "d": 86400}[m.group(2)]
    try:
        return float(text)
    except ValueError:
        return datetime.datetime.fromisoformat(text).replace(tzinfo=datetime.UTC).timestamp()


def print_results(rows, metric):
    import time

    for run, arm, e in rows:
        ms = e["metrics"]
        head = f"{ms[metric]:.4g}" if metric else ""
        rest = " ".join(f"{k}={v:.4g}" if isinstance(v, (int, float)) else f"{k}={v}" for k, v in ms.items()
                        if k != metric)
        extra = " | ".join(str(e[k]) for k in ("bundle", "vs", "verdict") if e.get(k))
        print(f"{time.strftime('%m-%d %H:%M', time.gmtime(e['t']))}  {arm or '?':10s} {run:58s} {head:>8s}  {rest}"
              + (f"  | {extra}" if extra else ""))


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("reserve")
    r.add_argument("run")
    for name in ("gate",):
        g = sub.add_parser(name)
    for p in (r, g):
        p.add_argument("--arm", required=True)
        p.add_argument("--platform", required=True)
        p.add_argument("--gpu", required=True)
        p.add_argument("--gpu-count", type=int, default=1)
        p.add_argument("--bound", type=float, required=True)
        p.add_argument("--long-run-24h", action="store_true")
        p.add_argument("--allowance-exempt", help="owner-approved reason: outside the arm's GPU allowance")
    r.add_argument("--max-seconds", type=int, required=True)
    r.add_argument("--auth", required=True)
    r.add_argument("--what", required=True)
    r.add_argument("--kind")
    r.add_argument("--volume", help='"run": the run gets its own Modal v2 volume arc-run-<run>; default the shared arc-prolong-runs')
    r.add_argument("--set", action="append")
    s = sub.add_parser("settle")
    s.add_argument("run")
    s.add_argument("state")
    s.add_argument("--cause", required=True)
    s.add_argument("--set", action="append")
    sa = sub.add_parser("settle-attempt", help="close or correct an earlier attempt's ledger row")
    sa.add_argument("run")
    sa.add_argument("n", type=int)
    sa.add_argument("state")
    sa.add_argument("--cause", required=True)
    sa.add_argument("--set", action="append")
    rq2 = sub.add_parser("require")
    rq2.add_argument("run")
    rn = sub.add_parser("running")
    rn.add_argument("run")
    rn.add_argument("--set", action="append")
    q = sub.add_parser("queue")
    q.add_argument("run")
    q.add_argument("--set", action="append", required=True)
    lg = sub.add_parser("ledger")
    lg.add_argument("run")
    lg.add_argument("--set", action="append", required=True)
    sub.add_parser("cpu-autoreg")
    cd = sub.add_parser("cpu-app-drop")
    cd.add_argument("app_id")
    c = sub.add_parser("cpu-app")
    c.add_argument("app_id")
    rs = sub.add_parser("result")
    rs.add_argument("run")
    rs.add_argument("--metric", action="append", required=True, help="name=number")
    rs.add_argument("--bundle")
    rs.add_argument("--vs")
    rs.add_argument("--verdict")
    rs.add_argument("--note")
    rq = sub.add_parser("results")
    rq.add_argument("--arm")
    rq.add_argument("--since")
    rq.add_argument("--metric")
    rq.add_argument("--prefix")
    rq.add_argument("--json", action="store_true")
    sh = sub.add_parser("show")
    sh.add_argument("run")
    ou = sub.add_parser("out")
    ou.add_argument("run")
    at = sub.add_parser("attach")
    at.add_argument("run")
    at.add_argument("log")
    pu = sub.add_parser("push")
    pu.add_argument("runs", nargs="+")
    vo = sub.add_parser("volume", help="the run's Modal volume (its record's, else arc-prolong-runs)")
    vo.add_argument("run")
    rt = sub.add_parser("retire", help="tools/ops/retire.py: archive + delete a run's own volume (by hand, on the "
                                       "explicit approval; never scheduled)")
    rt.add_argument("run")
    rt.add_argument("--archive", action="store_true")
    rt.add_argument("--delete", action="store_true")
    ra = sub.add_parser("rename-arm", help="rewrite every record's arm OLD -> NEW (records, ledger and queue rows)")
    ra.add_argument("old")
    ra.add_argument("new")
    ra.add_argument("--dry", action="store_true")
    for name in ("sync", "build", "migrate", "billing", "pull", "unclassified", "snapshot"):
        sub.add_parser(name)
    a = ap.parse_args()
    try:
        if a.cmd == "out":
            print(R.out_id(a.run))
        elif a.cmd == "attach":
            app = R.attach(a.run, Path(a.log).read_text(errors="replace") if Path(a.log).exists() else "")
            if not app:
                print(f"APP_ID NOT RECORDED {a.run}: no ap-... id in {a.log}", file=sys.stderr)
                sys.exit(1)
            print(f"record {a.run}: app_id {app}")
        elif a.cmd == "reserve":
            R.reserve(a.run, arm=a.arm, platform=a.platform, gpu=a.gpu, gpu_count=a.gpu_count, bound_usd=a.bound,
                      max_seconds=a.max_seconds, authorization=a.auth, what=a.what, kind=a.kind,
                      long_run_24h=a.long_run_24h, allowance_exempt=a.allowance_exempt, volume=a.volume, **kv(a.set))
            print(f"reserved {a.run} (${a.bound:.2f}) on volume {R.volume_of(a.run)}")
        elif a.cmd == "gate":
            for w in R.gate(a.arm, a.gpu, a.gpu_count, a.bound, platform=a.platform, long_run_24h=a.long_run_24h,
                            allowance_exempt=a.allowance_exempt):
                print(f"GATE WARNING: {w}")
            print("gate OK")
        elif a.cmd == "settle":
            R.settle(a.run, a.state, end_cause=a.cause, **kv(a.set))
        elif a.cmd == "settle-attempt":
            R.settle_attempt(a.run, a.n, a.state, end_cause=a.cause, **kv(a.set))
        elif a.cmd == "require":
            R.require_reservation(a.run)
            print(f"{a.run}: reserved, launch may go")
        elif a.cmd == "running":
            R.set_ledger(a.run, running=True, **kv(a.set))
        elif a.cmd == "unclassified":
            runs = R.unclassified()
            print("\n".join(runs) or "none")
            sys.exit(1 if runs else 0)
        elif a.cmd == "queue":
            R.set_queue(a.run, **kv(a.set))
        elif a.cmd == "ledger":
            R.set_ledger(a.run, **kv(a.set))
        elif a.cmd == "cpu-app-drop":
            R.drop_cpu_app(a.app_id)
            print(f"{a.app_id}: dropped")
        elif a.cmd == "cpu-autoreg":
            for aid, name in R.autoregister_cpu_apps():
                print(f"registered {aid} as {name}")
        elif a.cmd == "cpu-app":
            R.register_cpu_app(a.app_id)
        elif a.cmd == "result":
            ms = {k: float(v) for k, v in (m.split("=", 1) for m in a.metric)}
            R.add_result(a.run, ms, bundle=a.bundle, vs=a.vs, verdict=a.verdict, note=a.note)
            print(f"result recorded for {a.run}")
        elif a.cmd == "results":
            rows = R.results(arm=a.arm, since=since_epoch(a.since), metric=a.metric, prefix=a.prefix)
            if a.json:
                print(json.dumps([{"run": r, "arm": arm, **e} for r, arm, e in rows], indent=1))
            else:
                print_results(rows, a.metric)
        elif a.cmd == "show":
            print(json.dumps(R.get(a.run), indent=1))
        elif a.cmd == "push":
            push(a.runs)
        elif a.cmd == "retire":
            sys.path.insert(0, str(Path(__file__).resolve().parent))
            import retire
            retire.main([a.run] + (["--archive"] if a.archive else []) + (["--delete"] if a.delete else []))
        elif a.cmd == "volume":
            print(R.volume_of(a.run))
        elif a.cmd == "pull":
            pull()
        elif a.cmd == "snapshot":
            snapshot()
        elif a.cmd == "sync":
            dropped = R.sync()
            print("views regenerated" + (f"; DROPPED direct view edits of: {' '.join(dropped[:20])}" if dropped else ""))
        elif a.cmd == "build":
            R.write_views()
        elif a.cmd == "billing":
            usd = R.modal_billed(modal_bin=str(R.ROOT / ".venv/bin/modal"))
            R.set_modal_billed(usd)
            print(f"modal billed since 2026-09-01: ${usd:,.2f}")
        elif a.cmd == "rename-arm":
            runs = R.rename_arm(a.old, a.new, dry=a.dry)
            print(f"{'would rename' if a.dry else 'renamed'} {a.old} -> {a.new} in {len(runs)} records")
        elif a.cmd == "migrate":
            rules = json.loads((R.ROOT / "docs/html/arm-state.json").read_text()).get("run_arms", {})
            print(f"migrated {R.migrate(arm_rules=rules)} records")
    except (R.GateError, ValueError) as e:  # exit 1; the record is unchanged
        what = "GATE" if a.cmd in ("reserve", "gate", "require") else a.cmd.upper()
        sys.exit(f"{what} REFUSED (record unchanged): {e}")


if __name__ == "__main__":
    main()
