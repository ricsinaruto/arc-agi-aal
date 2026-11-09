"""One record file per run: migration round trip, gate (own allowance + global caps only), views regenerated (direct
view edits dropped and flagged), concurrent writers (same record, gate) across processes, settled snapshot."""
import json
import multiprocessing
from pathlib import Path

import pytest

from arc_infra.runops import records as R

QUEUE = {"descriptions": {"arena-x": "an old arena run", "rl1-loop-a": "RL leg a"},
         "queued": [],
         "live": [{"run": "rl1-loop-a", "arm": "rl", "platform": "modal", "gpu": "2 x B300", "state": "running",
                   "start_epoch": 100, "what": "RL leg a"},
                  {"run": "b2-tp-1", "arm": "throughput", "platform": "nebius", "gpu": "RTX PRO 6000",
                   "start_epoch": 50, "what": "tp"}],
         "next": ["RL: hero"], "cpu_apps": ["ap-1", "ap-2"], "cpu_app_names": ["arc-x"]}
SPEND = {"cap_usd": 30000, "infrastructure_reserve_usd": 2.0,
         "runs": [{"run_id": "arena-x", "state": "complete", "charged_bound_usd": 10.0},
                  {"run_id": "rl1-loop-a", "state": "reserved", "charged_bound_usd": 400.0, "start_epoch": 100}],
         "cap_history": [{"date": "2026-10-18"}],
         "nebius": {"cap_usd": 5000.0, "runs": [{"run_id": "b2-tp-1", "state": "reserved", "charged_bound_usd": 30.0}],
                    "note": "separate"}}
BUDGET = {"allowances": {"rl": {"usd": 5000, "gpus": {"RTX PRO 6000": 2}}, "sft": {"gpus": {"RTX PRO 6000": 2}},
                         "mfu": {"gpus": {"B300": 1}}, "arc3sub": {"usd": 2000, "gpus": {"RTX PRO 6000": 1}},
                         "envs": {"usd": 300, "gpus": {}}},
          "pools": {"b300-posttrain": {"kind": "B300", "arms": ["rl", "sft", "mfu"], "cap": 4}},
          "caps": {"gpus": {"B300": 8, "RTX PRO 6000": 8}, "usd": {"modal": 30000, "nebius": 5000}}}


@pytest.fixture
def root(tmp_path, monkeypatch):
    (tmp_path / "docs").mkdir()
    (tmp_path / "docs/modal-queue.json").write_text(json.dumps(QUEUE, indent=1) + "\n")
    (tmp_path / "docs/spend.json").write_text(json.dumps(SPEND, indent=1) + "\n")
    (tmp_path / "docs/budget.json").write_text(json.dumps(BUDGET))
    monkeypatch.delenv("ARC_RECORDS_NO_SYNC", raising=False)
    monkeypatch.setattr(R, "ROOT", tmp_path)  # never the real checkout
    R.migrate(tmp_path, {"rl1-": "rl", "b2-tp": "throughput"})
    return tmp_path


def views(root):
    return (json.loads((root / "docs/modal-queue.json").read_text()), json.loads((root / "docs/spend.json").read_text()))


def test_round_trip(root):
    assert R.build(root) == (QUEUE, SPEND)
    assert R.get("rl1-loop-a", root)["arm"] == "rl" and R.get("ap-1", root)["cpu_app"]["order"] == 0
    assert len(list((root / "docs/runs").glob("*.json"))) == 5


def test_gate_checks_own_allowance_and_global_caps(root):
    R.gate("rl", "2 x B300", 2, 100, platform="modal", root=root)  # pool: 2 live + 2 <= 4
    with pytest.raises(R.GateError, match="pool b300-posttrain has 2 B300 live"):
        R.gate("sft", "B300:3", 3, 100, platform="modal", root=root)
    with pytest.raises(R.GateError, match="its allowance 1"):
        R.gate("mfu", "2 x B300", 2, 1, platform="modal", root=root)  # its own cap inside the pool
    with pytest.raises(R.GateError, match="its allowance 0"):
        R.gate("envs", "B300", 1, 1, platform="modal", root=root)  # not in the pool
    with pytest.raises(R.GateError, match=r" allowance"):
        R.gate("rl", "B300", 1, 4700, platform="modal", root=root)
    with pytest.raises(R.GateError, match="its allowance 1"):  # b2-tp-1 is live
        R.gate("throughput", "RTX PRO 6000", 1, 10, platform="nebius", root=root)
    R.gate("throughput", "RTX PRO 6000", 1, 10, platform="nebius", root=root, allowance_exempt="c100 300K read")
    with pytest.raises(R.GateError, match="no allowance"):
        R.gate("nobody", "RTX PRO 6000", 1, 1, platform="nebius", root=root)
    b = {**BUDGET, "pools": {}, "allowances": {**BUDGET["allowances"], "rl": {"usd": 99999, "gpus": {"B300": 99}}}}
    with pytest.raises(R.GateError, match="global: 2 B300 live"):
        R.gate("rl", "7 x B300", 7, 1, platform="modal", root=root, budget={**b, "spend": {}})


def test_reserve_settle_and_views(root):
    R.reserve("rl1-loop-b", arm="rl", platform="modal", gpu="2 x B300", gpu_count=2, bound_usd=50, max_seconds=3600,
              authorization="test", what="RL leg b")
    q, s = views(root)
    assert [r["run"] for r in q["live"]] == ["rl1-loop-a", "b2-tp-1", "rl1-loop-b"]
    assert s["runs"][-1]["run_id"] == "rl1-loop-b" and s["runs"][-1]["state"] == "reserved"
    with pytest.raises(R.GateError, match="attempt 1 is still reserved"):
        R.reserve("rl1-loop-b", arm="rl", platform="modal", gpu="B300", bound_usd=1, max_seconds=1,
                  authorization="x", what="x")
    R.set_ledger("rl1-loop-b", app_id="ap-9")
    assert views(root)[1]["runs"][-1]["app_id"] == "ap-9"
    with pytest.raises(R.GateError, match="no ledger row"):
        R.set_ledger("nope-1", app_id="x")
    with pytest.raises(ValueError, match="end_cause"):
        R.settle("rl1-loop-b", "stopped", observed_seconds=60)
    R.settle("rl1-loop-b", "stopped", end_cause="stopped-owner", observed_seconds=60)
    q, s = views(root)
    assert "rl1-loop-b" not in [r["run"] for r in q["live"]] and s["runs"][-1]["state"] == "stopped"
    with pytest.raises(R.GateError, match="settled"):  # a closed row is corrected by settle --set, not patched
        R.set_ledger("rl1-loop-b", note="late")
    R.set_queue("rl1-loop-b", next="late edit")  # dots3-nb2, 24 Oct: must not re-open a bare live row
    assert "rl1-loop-b" not in [r["run"] for r in views(root)[0]["live"]]


def test_direct_view_edits_are_dropped_and_flagged(root):
    """Phase D (24 Oct): records.py is the only writer; a direct view edit is overwritten and reported."""
    R.sync(root)  # baseline
    q, s = views(root)
    s["nebius"]["runs"].append({"run_id": "b2-tp-2", "state": "reserved", "charged_bound_usd": 5.0})
    q["live"] = [r for r in q["live"] if r["run"] != "b2-tp-1"]
    (root / "docs/spend.json").write_text(json.dumps(s, indent=1))
    (root / "docs/modal-queue.json").write_text(json.dumps(q, indent=1))
    R.set_queue("rl1-loop-a", what="RL leg a, read every 4")  # a record write publishes the views
    q2, s2 = views(root)
    assert R.get("b2-tp-2", root) is None and "queue" in R.get("b2-tp-1", root)
    assert [r["run_id"] for r in s2["nebius"]["runs"]] == ["b2-tp-1"] and (q2, s2) == R.build(root)
    assert next(r for r in q2["live"] if r["run"] == "rl1-loop-a")["what"] == "RL leg a, read every 4"
    assert R.dropped_legacy(root) == ["b2-tp-1", "b2-tp-2"]
    assert R.sync(root) == []  # last-*.json track the regenerated views: an old-code sync imports nothing stale
    assert json.loads((root / "runs/ops/records/last-spend.json").read_text()) == s2


def test_views_are_created_in_a_fresh_checkout(root):
    (root / "docs/spend.json").unlink()
    (root / "docs/modal-queue.json").unlink()
    R.set_queue("rl1-loop-a", what="x")
    assert views(root) == R.build(root)


def _writer(args):
    root, i = args
    import os

    os.environ["ARC_RECORDS_NO_SYNC"] = "1"
    R.ROOT = Path(root)
    for j in range(20):
        R.set_queue(f"run-{i}", root=Path(root), **{f"k{j}": j})
    return i


def test_concurrent_writers_lose_nothing(root):
    with multiprocessing.get_context("spawn").Pool(4) as pool:
        pool.map(_writer, [(str(root), i) for i in range(4)])
    for i in range(4):
        assert len([k for k in R.get(f"run-{i}", root)["queue"]["row"] if k.startswith("k")]) == 20


def _same_record(args):
    root, i = args
    R.ROOT = Path(root)  # publish on: every write also takes the gate lock and regenerates the views
    for j in range(15):
        R.update("shared", lambda rec, i=i, j=j: rec.setdefault("log", []).append(f"{i}-{j}"), Path(root))
        R.publish(Path(root))
    return i


def _reserve_one(args):
    root, i = args
    R.ROOT = Path(root)
    try:
        R.reserve(f"rl-par-{i}", arm="rl", platform="nebius", gpu="RTX PRO 6000", bound_usd=1, max_seconds=60,
                  authorization="t", what="t", root=Path(root))
        return True
    except R.GateError:
        return False


def test_parallel_processes_serialize_on_one_record_and_the_gate(root):
    """24 Oct (records out of git): many worktree agents write at once; per-record flock and the gate lock serialize."""
    ctx = multiprocessing.get_context("spawn")
    with ctx.Pool(8) as pool:
        pool.map(_same_record, [(str(root), i) for i in range(8)])
        assert sum(pool.map(_reserve_one, [(str(root), i) for i in range(8)])) == 2  # rl's RTX allowance is 2
    assert sorted(R.get("shared", root)["log"]) == sorted(f"{i}-{j}" for i in range(8) for j in range(15))
    assert views(root) == R.build(root)


def test_snapshot_copies_only_ended_records(root):
    R.settle("b2-tp-1", "stopped", end_cause="completed")
    assert R.snapshot(root) == 2  # arena-x (complete) and b2-tp-1
    assert sorted(p.name for p in (root / "docs/runs-settled").iterdir()) == ["arena-x.json", "b2-tp-1.json"]
    assert R.snapshot(root) == 0


def test_root_from_worktree():
    here = R._default_root()
    assert not str(here).endswith("/infra") or ".claude/worktrees" not in str(here)


def test_reconcile_queue_ops(root):
    R.apply_queue_ops([("set", "rl1-loop-a", {"end": "12:00"}), ("drop", "b2-tp-1", {}),
                       ("add", "b2-new-1", {"run": "b2-new-1", "arm": "throughput", "platform": "nebius",
                                            "gpu": "RTX PRO 6000", "auto": True}),
                       ("add", "rl1-loop-a", {"run": "rl1-loop-a", "what": "must not replace"})], root=root)
    q, _ = views(root)
    assert [r["run"] for r in q["live"]] == ["rl1-loop-a", "b2-new-1"]
    assert q["live"][0]["end"] == "12:00" and q["live"][0]["what"] == "RL leg a"
    assert R.get("b2-new-1", root)["arm"] == "throughput"


def test_modal_gate_uses_billed_total(root):
    b = {**json.loads((root / "docs/budget.json").read_text()), "spend": {"infrastructure_reserve_usd": 2.0},
         "caps": {"usd": {"modal": 1000}}, "modal_billed": {"usd": 500.0}}
    R.gate("rl", "RTX PRO 6000", 1, 99, platform="modal", root=root, budget=b)  # 500 billed + 400 open + 99
    with pytest.raises(R.GateError, match=r" committed"):
        R.gate("rl", "RTX PRO 6000", 1, 101, platform="modal", root=root, budget=b)


def test_arm_for_longest_prefix():
    rules = {"b2-": "throughput", "b2-mfu": "mfu", "rl": "rl", "rl1-loop-xpf": "envs"}
    assert R.arm_for("b2-mfu2-rtxmark-v9", rules) == "mfu" and R.arm_for("b2-sabl-x", rules) == "arc3sub"
    assert R.arm_for("rl1-loop-xpf-1", rules) == "envs" and R.arm_for("zzz", rules) is None


def test_usd_warn_mode(root):
    b = json.loads((root / "docs/budget.json").read_text())
    b["allowances"]["arc3sub"] = {"usd": 100, "usd_mode": "warn", "gpus": {"RTX PRO 6000": 2}}
    assert R.gate("throughput", "RTX PRO 6000", 1, 5, platform="nebius", root=root, budget=b) == []
    w = R.gate("throughput", "RTX PRO 6000", 1, 80, platform="nebius", root=root, budget=b)
    assert len(w) == 1 and "warn-only" in w[0]
    b["allowances"]["arc3sub"]["gpus"] = {"RTX PRO 6000": 1}
    with pytest.raises(R.GateError, match="its allowance 1"):  # GPUs stay enforced
        R.gate("throughput", "RTX PRO 6000", 1, 50, platform="nebius", root=root, budget=b)


def test_pull_and_snapshot_with_untracked_records(tmp_path, monkeypatch):
    """Records and views are untracked: a pull never conflicts on them; the daily snapshot commits settled records."""
    import importlib.util
    import subprocess

    def git(cwd, *a):
        return subprocess.run(["git", "-C", str(cwd), *a], capture_output=True, text=True, check=True).stdout

    spec = importlib.util.spec_from_file_location("ops_records", Path(__file__).resolve().parents[1] / "tools/ops/records.py")
    cli = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(cli)
    origin, a, b = tmp_path / "origin.git", tmp_path / "a", tmp_path / "b"
    subprocess.run(["git", "init", "-q", "--bare", "-b", "main", str(origin)], check=True)
    (a / "docs").mkdir(parents=True)
    (a / "docs/modal-queue.json").write_text(json.dumps(QUEUE, indent=1) + "\n")
    (a / "docs/spend.json").write_text(json.dumps(SPEND, indent=1) + "\n")
    (a / "docs/budget.json").write_text(json.dumps(BUDGET))
    monkeypatch.setattr(R, "ROOT", a)
    monkeypatch.delenv("ARC_RECORDS_NO_SYNC", raising=False)
    R.migrate(a, {"rl1-": "rl", "b2-tp": "throughput"})
    (a / ".gitignore").write_text("runs/\ndocs/spend.json\ndocs/modal-queue.json\ndocs/runs-settled/\n")
    for c in (["init", "-q", "-b", "main"], ["config", "user.email", "t@t"], ["config", "user.name", "t"],
              ["add", "-A"], ["commit", "-q", "-m", "init"], ["remote", "add", "origin", str(origin)],
              ["push", "-q", "origin", "main"]):
        git(a, *c)
    assert "docs/runs/" not in git(a, "ls-files") and "spend.json" not in git(a, "ls-files")
    git(tmp_path, "clone", "-q", str(origin), str(b))
    (b / "x.txt").write_text("other work")
    git(b, "add", "x.txt")
    git(b, "-c", "user.email=t@t", "-c", "user.name=t", "commit", "-q", "-m", "other")
    git(b, "push", "-q", "origin", "main")
    R.set_queue("rl1-loop-a", what="edited here")
    R.settle("b2-tp-1", "stopped", end_cause="completed")
    cli.pull()
    assert (a / "x.txt").is_file() and R.get("rl1-loop-a", a)["queue"]["row"]["what"] == "edited here"
    cli.snapshot()
    assert git(a, "status", "--porcelain") == ""
    assert sorted(p.name for p in (a / "docs/runs-settled").iterdir()) == ["arena-x.json", "b2-tp-1.json"]
    assert git(a, "ls-files", "docs/runs-settled") == ""
    assert git(a, "rev-parse", "HEAD") == git(a, "rev-parse", "origin/main")


def test_exempt_flag_on_the_ledger_row(root):
    R.update("b2-tp-1", lambda rec: rec["ledger"][-1]["row"].update(allowance_exempt="c100 300K read"), root)
    R.gate("throughput", "RTX PRO 6000", 1, 10, platform="nebius", root=root)  # its live read no longer counts


def test_results_block_and_query(root):
    R.add_result("b2-tp-1", {"score300k": 13.7, "tok_s": 1534}, bundle="v85p@abc", vs="b2-tp-0", verdict="flat", t=1000)
    R.add_result("rl1-loop-a", {"joint": 0.41}, t=2000)
    R.add_result("b2-tp-1", {"score300k": 14.1}, t=3000)
    got = R.results(metric="score300k", root=root)
    assert [(r, e["metrics"]["score300k"]) for r, _, e in got] == [("b2-tp-1", 13.7), ("b2-tp-1", 14.1)]
    assert [r for r, _, _ in R.results(arm="rl", root=root)] == ["rl1-loop-a"]
    assert len(R.results(since=1500, root=root)) == 2
    assert views(root)[1]["nebius"]["runs"][0]["run_id"] == "b2-tp-1"  # results stay out of the legacy views


def test_lifecycle_causes_and_attempts(root):
    run = "rl1-loop-c"
    R.reserve(run, arm="rl", platform="modal", gpu="B300", bound_usd=5, max_seconds=60, authorization="t", what="c")
    assert R.state_of(R.get(run, root)) == "reserved"
    R.set_ledger(run, app_id="ap-c1")  # a real app id: running, also in the ledger view
    assert R.get(run, root)["state"] == "running" and views(root)[1]["runs"][-1]["state"] == "running"
    assert R.get(run, root)["ledger"][-1]["row"]["state"] == "reserved"  # the launcher's own row is untouched
    for bad in (None, "done", "bug:", "bug:Has Space"):
        with pytest.raises(ValueError):
            R.settle(run, "stopped", end_cause=bad)
    R.settle(run, "stopped", end_cause="bug:nccl-hang")
    with pytest.raises(R.GateError, match="no such run"):
        R.settle(run + "-typo", "stopped", end_cause="completed")
    assert R.get(run + "-typo", root) is None  # no stray record
    rec = R.get(run, root)
    assert (rec["state"], rec["end_cause"], rec["ledger"][-1]["row"]["end_cause"]) == ("ended", "bug:nccl-hang",
                                                                                         "bug:nccl-hang")
    R.reserve(run, arm="rl", platform="modal", gpu="B300", bound_usd=5, max_seconds=60, authorization="t", what="c")
    rec = R.get(run, root)
    assert rec["attempt"] == 2 and rec["state"] == "reserved" and "end_cause" not in rec
    assert [e["row"]["run_id"] for e in rec["ledger"]] == [run, f"{run}/attempt-2"]
    assert [(a["n"], a["state"], a.get("end_cause")) for a in rec["attempts"]] == [(1, "ended", "bug:nccl-hang"),
                                                                                  (2, "reserved", None)]
    q, s = views(root)
    assert [r for r in q["live"] if r["run"] == run][0]["attempt"] == 2
    assert [r["run_id"] for r in s["runs"]][-2:] == [run, f"{run}/attempt-2"]


def test_reconcile_marks_running_and_unsettled_drop_unclassified(root):
    R.apply_queue_ops([("set", "rl1-loop-a", {"start": "01:00"})], root)
    assert R.get("rl1-loop-a", root)["state"] == "running"
    R.apply_queue_ops([("drop", "rl1-loop-a", {})], root)
    rec = R.get("rl1-loop-a", root)
    assert (rec["state"], rec["end_cause"]) == ("ended", "unclassified")
    assert R.unclassified(root) == ["rl1-loop-a"]
    R.settle("rl1-loop-a", "stopped", end_cause="preempt")  # the arm states the cause later
    assert R.unclassified(root) == [] and R.get("rl1-loop-a", root)["end_cause"] == "preempt"


def test_a_dropped_attempt_is_closed_by_the_next_reserve_or_settle_attempt(root):
    run = "rl1-loop-d"
    args = dict(arm="rl", platform="modal", gpu="B300", bound_usd=5, max_seconds=60, authorization="t", what="d")
    R.reserve(run, **args)
    R.apply_queue_ops([("drop", run, {})], root)  # gone without a settle: the row stays open for a late settle
    assert R.get(run, root)["ledger"][-1]["row"]["state"] == "reserved"
    R.reserve(run, **args)
    rec = R.get(run, root)
    first = rec["ledger"][0]["row"]
    assert (first["state"], first["end_cause"]) == ("ended", "unclassified") and first["end_epoch"]
    assert rec["ledger"][-1]["row"]["state"] == "reserved" and rec["attempts"][0]["end_epoch"]
    R.settle(run, "stopped", end_cause="completed")
    R.reserve(run, **args)
    # the correction: the earlier row only; the current attempt, record and queue row stay
    R.settle_attempt(run, 1, "stopped", end_cause="preempt", end_epoch=123.4, note="last feed row")
    rec = R.get(run, root)
    assert (rec["ledger"][0]["row"]["state"], rec["ledger"][0]["row"]["end_cause"]) == ("stopped", "preempt")
    assert rec["ledger"][0]["row"]["end_epoch"] == 123 and rec["ledger"][0]["row"]["note"] == "last feed row"
    assert (rec["attempts"][0]["state"], rec["attempts"][0]["end_cause"], rec["attempts"][0]["end_epoch"]) == ("ended", "preempt", 123)
    assert rec["state"] == "reserved" and rec["attempt"] == 3 and rec["queue"]["section"] == "live"
    assert rec["ledger"][-1]["row"]["state"] == "reserved"
    with pytest.raises(R.GateError, match="not an earlier attempt"):
        R.settle_attempt(run, 3, "stopped", end_cause="preempt")
    with pytest.raises(R.GateError, match="no attempt 0"):
        R.settle_attempt(run, 0, "stopped", end_cause="preempt")
    with pytest.raises(ValueError):
        R.settle_attempt(run, 1, "stopped", end_cause="nope")


def test_backfill_states(root):
    assert R.backfill_states(root, causes={"arena-x": "bug:renderer"}) == 3
    got = {r["run"]: (r.get("state"), r.get("end_cause")) for r in R.records(root) if r.get("ledger")}
    assert got == {"arena-x": ("ended", "bug:renderer"), "rl1-loop-a": ("running", None),
                   "b2-tp-1": ("running", None)}
    assert R.backfill_states(root) == 0


def test_refused_reserve_blocks_the_launch(root):
    """24 Oct: MFU-1's launch ran 40 s after records.py printed a refusal. reserve exits non-zero and leaves no
    reservation, and require_reservation (every launcher, before its cloud call) refuses."""
    import subprocess
    import sys

    cli = [sys.executable, str(Path(__file__).resolve().parents[1] / "tools/ops/records.py")]
    env = {"ARC_RECORDS_ROOT": str(root), "ARC_RECORDS_NO_SYNC": "1", "PATH": "/usr/bin:/bin",
           "PYTHONPATH": str(Path(__file__).resolve().parents[1] / "src")}
    r = subprocess.run(cli + ["reserve", "mfu-x", "--arm", "mfu", "--platform", "modal", "--gpu", "3 x B300",
                              "--gpu-count", "3", "--bound", "5", "--max-seconds", "60", "--auth", "t", "--what", "t"],
                       capture_output=True, text=True, env=env)
    assert r.returncode != 0 and "GATE REFUSED" in r.stderr and "its allowance 1" in r.stderr
    with pytest.raises(R.GateError, match="no reservation"):
        R.require_reservation("mfu-x", root)
    r = subprocess.run(cli + ["require", "mfu-x"], capture_output=True, text=True, env=env)
    assert r.returncode != 0 and "no reservation" in r.stderr
    R.reserve("mfu-y", arm="mfu", platform="modal", gpu="B300", bound_usd=1, max_seconds=60, authorization="t",
              what="t")
    assert R.require_reservation("mfu-y", root)["arm"] == "mfu"
    R.set_ledger("mfu-y", app_id="ap-1")  # launched: a second launch on the same reservation is refused
    with pytest.raises(R.GateError, match="is running, not reserved"):
        R.require_reservation("mfu-y", root)


def test_settled_modal_row_costs_its_seconds_not_its_bound(root):
    """RL 24 Oct: a settle without an actual kept the full bound ("no-actual") and blocked the arm's $ gate."""
    R.reserve("rl1-z", arm="rl", platform="modal", gpu="2 x B300", gpu_count=2, bound_usd=400, max_seconds=84000,
              authorization="t", what="t")
    row = R.get("rl1-z", root)["ledger"][-1]["row"]
    assert row["gpu_rate_per_second"] == 0.001972
    R.settle("rl1-z", "stopped", end_cause="superseded", end_epoch=row["start_epoch"] + 1000)
    cost, kind = R.row_cost(R.get("rl1-z", root)["ledger"][-1]["row"])
    assert kind == "derived" and abs(cost - 1000 * 2 * 0.001972) < 1e-6
    old = {"state": "stopped", "platform": "modal", "gpu": "B300:2", "gpu_count": 2, "charged_bound_usd": 400,
           "start_epoch": 100, "end_epoch": 600}  # a pre-24-Oct row without a rate: the table fills it
    assert R.row_cost(old) == (500 * 2 * 0.001972, "derived")


def test_settle_is_terminal_against_a_stale_view(root):
    """24 Oct: PTH's settle of c7g was reverted by a sync that imported an older view. Since phase D views are never
    imported: a stale view is overwritten and the settle stands."""
    R.settle("rl1-loop-a", "stopped", end_cause="bug:teacher-backlog-order")
    q, s = views(root)
    q["live"].append({"run": "rl1-loop-a", "arm": "rl", "platform": "modal", "gpu": "2 x B300", "state": "running"})
    next(r for r in s["runs"] if r["run_id"] == "rl1-loop-a").update(state="reserved")
    (root / "docs/modal-queue.json").write_text(json.dumps(q, indent=1) + "\n")
    (root / "docs/spend.json").write_text(json.dumps(s, indent=1) + "\n")
    R.sync(root)
    rec = R.get("rl1-loop-a", root)
    assert (rec["state"], rec["end_cause"]) == ("ended", "bug:teacher-backlog-order") and "queue" not in rec
    assert rec["ledger"][-1]["row"]["state"] == "stopped"
    assert "rl1-loop-a" not in [r["run"] for r in views(root)[0]["live"]]


def test_cause_case_and_refused_settle_exit(root, monkeypatch, capsys):
    """24 Oct: dots3's settle with an upper-case cause was refused and its background chain never saw it."""
    import importlib.util
    import subprocess
    import sys

    R.settle("b2-tp-1", "stopped", end_cause="Bug:Teacher-OOM")
    assert R.get("b2-tp-1", root)["end_cause"] == "bug:teacher-oom"
    cli = Path(__file__).resolve().parents[1] / "tools/ops/records.py"
    env = {**__import__("os").environ, "ARC_RECORDS_ROOT": str(root), "PYTHONPATH": str(cli.parents[2] / "src")}
    p = subprocess.run([sys.executable, str(cli), "settle", "nope-9", "stopped", "--cause", "completed"],
                       capture_output=True, text=True, env=env)
    assert p.returncode == 1 and "SETTLE REFUSED (record unchanged)" in p.stderr and not (root / "docs/runs/nope-9.json").exists()
    p = subprocess.run([sys.executable, str(cli), "settle", "b2-tp-1", "stopped", "--cause", "oops"],
                       capture_output=True, text=True, env=env)
    assert p.returncode == 1 and "end_cause" in p.stderr


def test_regenerated_view_is_not_a_dropped_edit(root):
    """A view rewritten from the records by another writer (no baseline update) is not flagged."""
    R.sync(root)
    R.update("ap-9", lambda rec: rec.update(cpu_app={"order": R._now_order()}), root)  # a record write, no publish
    q, s = R.build(root)
    (root / "docs/modal-queue.json").write_text(json.dumps(q, indent=1) + "\n")  # someone regenerates the view
    assert R.sync(root) == [] and R.dropped_legacy(root) == []


def test_modal_slots_cap_ten_gpus_for_every_arm(root):
    """Modal runs 10 GPUs at once, all types and arms; no arm holds a reserve."""
    R.update("sft-c7h", lambda rec: rec.update(arm="sft", queue={"section": "live", "order": 1, "row": {
        "run": "sft-c7h", "platform": "modal", "gpu": "2 x B300", "app_id": "ap-pth"}}), root)
    live = {"ap-pth": {"gpus": {"B300": 2}}, "ap-rl": {"gpus": {"B300": 2}}, "ap-x": {"gpus": {"H200": 3}}, "ap-cpu": {"gpus": {}}}
    assert R.modal_slots(R.records(root), live) == 7
    assert R.modal_slots([{"queue": {"section": "live", "row": {"platform": "modal", "gpu": "2 x B300"}}},
                          {"queue": {"section": "live", "row": {"platform": "nebius", "gpu": "RTX PRO 6000"}}}]) == 2
    for arm in ("mfu", "rl", "sft"):
        R.gate(arm, "B300", 3, 1, platform="modal", root=root, modal=live, allowance_exempt="t")  # 7 + 3 = 10
        with pytest.raises(R.GateError, match="modal slots: 7 GPUs live on Modal"):
            R.gate(arm, "B300", 4, 2, platform="modal", root=root, modal=live, allowance_exempt="t")
    R.update("dots3-h", lambda rec: rec.update(arm="dots3", state="reserved", ledger=[{"block": "", "order": 1, "row": {
        "run_id": "dots3-h", "state": "reserved", "start_epoch": __import__("time").time()}}],
        queue={"section": "live", "order": 2, "row": {"run": "dots3-h", "platform": "modal", "gpu": "3 x H200"}}), root)
    R.require_reservation("dots3-h", root, modal=live)  # 7 + 3 <= 10
    live["ap-y"] = {"gpus": {"H200": 1}}
    with pytest.raises(R.GateError, match="modal slots"):  # require re-checks right before the launch: 8 + 3 > 10
        R.require_reservation("dots3-h", root, modal=live)


def test_cpu_apps_without_a_row_are_registered(root):
    R.update("sft-c7h", lambda rec: rec.update(arm="sft", queue={"section": "live", "order": 1, "row": {
        "run": "sft-c7h", "platform": "modal", "gpu": "2 x B300", "app_id": "ap-pth"}}), root)
    live = {"ap-pth": {"name": "heal", "gpus": {"B300": 2}}, "ap-cpu": {"name": "dots3-prune", "gpus": {}},
            "ap-gpu": {"name": "x", "gpus": {"H200": 1}}}
    assert R.autoregister_cpu_apps(root, live) == [("ap-cpu", "auto:dots3-prune")]
    assert R.autoregister_cpu_apps(root, live) == []  # idempotent
    assert "ap-cpu" in R.build(root)[0]["cpu_apps"] and "ap-gpu" not in R.build(root)[0]["cpu_apps"]


def test_modal_billed_is_an_untracked_view(root):
    before = (root / "docs/budget.json").read_text()
    R.set_modal_billed(123.4, root)
    assert (root / "docs/budget.json").read_text() == before  # budget.json is never rewritten
    assert R.load_budget(root)["modal_billed"]["usd"] == 123.4


def test_a_worktree_root_maps_to_main(root):
    wt = root / ".claude" / "worktrees" / "arm"
    wt.mkdir(parents=True)
    assert R.runs_dir(wt) == root / "docs" / "runs" and R.budget_path(wt) == root / "docs" / "budget.json"


def test_drop_cpu_app_only_drops_cpu_only_records(root):
    R.register_cpu_app("ap-typo", root=root)
    R.drop_cpu_app("ap-typo", root)
    assert R.get("ap-typo", root) is None
    with pytest.raises(R.GateError, match="not a CPU-app-only record"):
        R.drop_cpu_app("rl1-loop-a", root)


def test_queue_and_ledger_share_app_id(root):
    """24 Oct: `queue --set app_id=` left the ledger row without it (m1gp2, tput10 read "no queue row")."""
    R.reserve("mfu-q1", arm="mfu", platform="modal", gpu="B300", bound_usd=1, max_seconds=60, authorization="t", what="q")
    R.set_queue("mfu-q1", app_id="ap-q1")
    rec = R.get("mfu-q1", root)
    assert rec["ledger"][-1]["row"]["app_id"] == "ap-q1" and rec["queue"]["row"]["app_id"] == "ap-q1"
    assert rec["state"] == "running"
    R.set_ledger("mfu-q1", app_id="ap-q2")
    rec = R.get("mfu-q1", root)
    assert rec["queue"]["row"]["app_id"] == "ap-q2" and rec["ledger"][-1]["row"]["app_id"] == "ap-q2"
    R.reserve("rl-q2", arm="rl", platform="modal", gpu="B300", bound_usd=1, max_seconds=60, authorization="t", what="q")
    R.set_queue("rl-q2", state="running")
    assert R.get("rl-q2", root)["state"] == "running"
    R.settle("mfu-q1", "stopped", end_cause="completed")
    R.set_queue("mfu-q1", app_id="ap-late")  # an ended run is not revived through the shared setter
    assert R.get("mfu-q1", root)["state"] == "ended" and R.get("mfu-q1", root)["ledger"][-1]["row"]["app_id"] == "ap-q2"


def test_out_id_follows_the_attempt(root):
    R.reserve("rl-o1", arm="rl", platform="modal", gpu="B300", bound_usd=1, max_seconds=60, authorization="t", what="o")
    assert R.out_id("rl-o1", root) == "rl-o1" and R.out_id("never-reserved", root) == "never-reserved"
    R.settle("rl-o1", "stopped", end_cause="completed")
    R.reserve("rl-o1", arm="rl", platform="modal", gpu="B300", bound_usd=1, max_seconds=60, authorization="t", what="o")
    assert R.out_id("rl-o1", root) == "rl-o1-attempt2"


def test_likely_owner_names_the_open_record_without_an_app(root):
    """25 Oct: sft-stream-check-eosA-v1 ran as arc-rl1 with no app_id; gpu_reconcile now names the likely owner."""
    import time as _t
    now = _t.time()
    R.reserve("sft-sc-a", arm="sft", platform="modal", gpu="B300", bound_usd=1, max_seconds=60, authorization="t", what="a")
    assert R.likely_owner("arc-rl1", now + 30, root) == "sft-sc-a"
    assert R.likely_owner("someone-else", now + 30, root) is None  # not an arc-* app
    assert R.likely_owner("arc-rl1", now + 7200, root) is None  # reserved too long before the app
    R.set_ledger("sft-sc-a", app_id="ap-named")
    assert R.likely_owner("arc-rl1", now + 30, root) is None  # it names its app now


def test_attach_records_the_app_from_a_launch_log(root):
    R.reserve("sft-sc-b", arm="sft", platform="modal", gpu="B300", bound_usd=1, max_seconds=60, authorization="t", what="b")
    assert R.attach("sft-sc-b", "Error: no app\n", root) is None
    out = "✓ Initialized. View run at https://modal.com/apps/arc-agi-3/main/ap-Xx0qvyo7LX4FWDnMC5EXeD\n"
    assert R.attach("sft-sc-b", out, root) == "ap-Xx0qvyo7LX4FWDnMC5EXeD"
    rec = R.get("sft-sc-b", root)
    assert rec["ledger"][-1]["row"]["app_id"] == rec["queue"]["row"]["app_id"] == "ap-Xx0qvyo7LX4FWDnMC5EXeD"


def test_cli_set_refuses_anything_but_key_value():
    import importlib.util
    spec = importlib.util.spec_from_file_location("records_cli", Path(__file__).resolve().parents[1] / "tools/ops/records.py")
    cli = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(cli)
    assert cli.kv(["app_id=ap-1", 'k={"a": 1}', "n=2"]) == {"app_id": "ap-1", "k": {"a": 1}, "n": 2}
    for bad in ['{"end_cause": "stopped-owner"}', "noequals", "=x"]:
        with pytest.raises(SystemExit):
            cli.kv([bad])


def test_build_between_syncs_is_not_a_dropped_edit(root, monkeypatch):
    """records.py build (write_views) refreshes sync's baseline: a later record write + sync flags nothing (28 Oct:
    sft-layers-p75-v1 flagged after a launch's build and running set)."""
    R.sync(root)
    monkeypatch.setenv("ARC_RECORDS_NO_SYNC", "1")
    R.set_queue("rl1-loop-a", what="first")
    R.write_views(root)
    R.set_queue("rl1-loop-a", what="second")
    monkeypatch.delenv("ARC_RECORDS_NO_SYNC")
    assert R.sync(root) == [] and R.dropped_legacy(root) == []


def test_record_write_during_sync_is_not_a_dropped_edit(root, monkeypatch):
    """28 Oct 13:18: a settle landed between sync's view build and its baseline build (update() takes only the run's
    lock); view != baseline flagged rl1-loop-seq-real-v10 as a dropped direct edit. One build feeds both now."""
    R.sync(root)
    monkeypatch.setenv("ARC_RECORDS_NO_SYNC", "1")
    R.set_queue("rl1-loop-a", what="first")
    monkeypatch.delenv("ARC_RECORDS_NO_SYNC")
    real_build, calls = R.build, []

    def build_then_write(*a, **k):  # the concurrent writer lands right after the second build of this sync
        out = real_build(*a, **k)
        calls.append(1)
        if len(calls) == 2:
            monkeypatch.setenv("ARC_RECORDS_NO_SYNC", "1")
            R.set_queue("rl1-loop-a", what="settled meanwhile")
            monkeypatch.delenv("ARC_RECORDS_NO_SYNC")
        return out

    monkeypatch.setattr(R, "build", build_then_write)
    R.sync(root)
    monkeypatch.setattr(R, "build", real_build)
    assert R.sync(root) == [] and R.dropped_legacy(root) == []


def test_per_run_volume_is_kept_across_attempts(root, monkeypatch):
    """one Modal volume per run (arc-run-<run>); attempts keep it; a record without one is on the shared volume."""
    from arc_infra.runops import volfetch as vf

    R.reserve("rl1-loop-v", arm="rl", platform="modal", gpu="B300", bound_usd=5, max_seconds=600, authorization="t",
              what="per-run volume", volume="run")
    assert R.volume_of("rl1-loop-v") == "arc-run-rl1-loop-v" and R.volume_of("rl1-loop-a") == R.SHARED_VOLUME
    R.settle("rl1-loop-v", "stopped", end_cause="preempt")
    R.reserve("rl1-loop-v", arm="rl", platform="modal", gpu="B300", bound_usd=5, max_seconds=600, authorization="t",
              what="attempt 2")
    assert R.volume_of("rl1-loop-v") == "arc-run-rl1-loop-v"
    long = "rl1-" + "x" * 80
    assert len(R.run_volume_name(long)) <= 64 and R.run_volume_name(long) != R.run_volume_name(long + "y")
    monkeypatch.setattr(R, "get", lambda run, root=None: {"rl1-loop-v": {"volume": "arc-run-rl1-loop-v"}}.get(run))  # volfetch reads the real ROOT
    assert vf.route("arc-prolong-runs", "validation/rl1-loop-v-attempt2/live/sampler.jsonl") == "arc-run-rl1-loop-v"
    assert vf.route("arc-prolong-runs", "validation/other/live/x") == "arc-prolong-runs"
    assert vf.route("arc-prolong-runs", "validation/preflight/rl1-loop-vNK/x") == "arc-prolong-runs"
    assert vf.mirror("arc-prolong-runs", "validation/rl1-loop-v/live/x").parts[-5] == "arc-run-rl1-loop-v"


def test_external_runs_are_never_counted():
    from arc_infra.runops import records as R
    b = {"external_runs": {"ids": ["arc2-x-twin-1"], "prefixes": ["arc2-rtx-c240-hx-twin-"]}}
    assert R.is_external("arc2-x-twin-1", b) and R.is_external("arc2-rtx-c240-hx-twin-p9", b)
    assert not R.is_external("arc2-rtx-c240-hx-conv1-p1", b) and not R.is_external("arc2-x-twin-1", {})


def test_rename_arm_keeps_the_live_count(root):
    """throughput -> arc3sub. Before and after the rewrite the live row counts under arc3sub."""
    with pytest.raises(R.GateError, match="its allowance 1"):
        R.gate("arc3sub", "RTX PRO 6000", 1, 10, platform="nebius", root=root)  # old name on the record, alias
    assert R.rename_arm("throughput", "arc3sub", root, dry=True) == ["b2-tp-1"]
    assert R.get("b2-tp-1", root)["arm"] == "throughput"  # dry run writes nothing
    assert R.rename_arm("throughput", "arc3sub", root) == ["b2-tp-1"]
    rec = R.get("b2-tp-1", root)
    rows = [rec, rec["queue"]["row"]] + [e["row"] for e in rec["ledger"]]
    assert all(r.get("arm") in (None, "arc3sub") for r in rows) and rec["arm"] == "arc3sub"
    with pytest.raises(R.GateError, match="its allowance 1"):
        R.gate("arc3sub", "RTX PRO 6000", 1, 10, platform="nebius", root=root)
    assert R.rename_arm("throughput", "arc3sub", root) == []


def test_calibration_stays_with_its_attempt(tmp_path):
    root = tmp_path
    (root / "docs/runs").mkdir(parents=True)
    (root / "docs/runs/r.json").write_text(json.dumps({"run": "r", "attempts": [{"n": 1}, {"n": 2}]}))
    R.set_calibration("r", {"bench": {"gpu_core_gflops": 1.0}}, root=root)
    rec = R.get("r", root=root)
    assert "calibration" not in rec["attempts"][0] and R.calibration(rec)["bench"]["gpu_core_gflops"] == 1.0
    (root / "docs/runs/s.json").write_text(json.dumps({"run": "s"}))
    R.set_calibration("s", {"bench": {}, "x": 1}, root=root)
    assert R.calibration(R.get("s", root=root))["x"] == 1


def test_stale_open_records_close_only_when_nothing_runs_behind_them(tmp_path, monkeypatch):
    """47 reserve rows were open for weeks; reconcile now closes what no app/VM runs behind."""
    monkeypatch.setenv("ARC_RECORDS_NO_SYNC", "1")
    (tmp_path / "docs/runs").mkdir(parents=True)
    now = 1_800_000_000.0

    def put(run, **rec):
        (tmp_path / f"docs/runs/{run}.json").write_text(json.dumps({"run": run, **rec}))

    row = lambda **r: [{"block": "", "order": 13, "row": {"run_id": r.pop("run_id", None), "state": "reserved", **r}}]
    put("legacy", ledger=row(max_seconds=600), state="reserved")  # migrated, no times
    put("old", ledger=row(max_seconds=600, start_epoch=now - 5 * 3600), state="reserved")
    put("young", ledger=row(max_seconds=3600, start_epoch=now - 3600), state="reserved")  # within its reservation
    put("vm", ledger=row(max_seconds=600, start_epoch=now - 9 * 3600), state="running")  # its VM is listed
    put("app", ledger=row(max_seconds=600, start_epoch=now - 9 * 3600, app_id="ap-x"), state="running")
    put("queued", ledger=row(max_seconds=600), state="reserved", queue={"section": "live", "row": {}})
    put("done", ledger=row(max_seconds=600, start_epoch=now - 9 * 3600) and
        [{"block": "", "order": 1, "row": {"state": "completed"}}], state="ended")
    put("dropped", ledger=row(max_seconds=600, start_epoch=now - 9 * 3600), state="ended", end_cause="unclassified",
        dropped_epoch=now - 4 * 3600)
    put("dropped-new", ledger=row(max_seconds=600), state="ended", dropped_epoch=now - 600)
    put("cpu", cpu_app={"name": "x"})
    recs = R.records(root=tmp_path)
    stale = R.stale_open(recs, live_runs={"vm"}, live_apps={"ap-x"}, now=now)
    assert sorted(stale) == ["dropped", "legacy", "old"]
    R.close_stale(stale, note="t", root=tmp_path)
    got = {r["run"]: r for r in R.records(root=tmp_path)}
    for run in stale:
        assert R.state_of(got[run]) == "ended" and got[run]["ledger"][-1]["row"]["state"] == "reconciled"
        assert R.row_cost(got[run]["ledger"][-1]["row"])[1] != "open"
    assert got["old"]["ledger"][-1]["row"]["end_epoch"] == round(now - 5 * 3600 + 600)
    assert got["dropped"]["ledger"][-1]["row"]["end_epoch"] == now - 4 * 3600
    assert R.stale_open(R.records(root=tmp_path), live_runs={"vm"}, live_apps={"ap-x"}, now=now) == []
