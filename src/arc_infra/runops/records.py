"""One record file per run: docs/runs/<run>.json replaces the shared docs/modal-queue.json and
docs/spend.json, which are views generated from the records (``build``) so mon.py, train_dash.py, gpu_reconcile.py and
the HTML pages keep reading them. Writers touch only their run's file under its own flock (no lost updates).
Caps and per-arm allowances live in docs/budget.json.

Out of git (24 Oct, phase D + item 4): the records and both views are machine-local and untracked in the main checkout
(every agent's ROOT); records.py is the only writer. A direct edit of a view is overwritten at the next write and
flagged on the dashboard (``dropped_legacy``). ``snapshot`` copies ended records to docs/runs-settled/ (tracked; the dash
loop commits it daily).

Record: {"run", "arm", "state", "end_cause", "attempt", "attempts": [...], "ledger": [{"block", "order", "row"}],
         "queue": {"section", "order", "row"}, "description": {"order", "text"}, "cpu_app": {"order"},
         "plan": {...watcher plan}, "results": [...]}
``state``: reserved -> running -> ended (``state_of`` derives it for records written before 24 Oct). ``end_cause`` is
required at settle: one of END_CAUSES or ``bug:<class>``; a run whose app/VM vanished without a settle is ended
"unclassified" by the dash pass and flagged until its arm settles it. A relaunch of an ended run is attempt N+1 of the
same record (``<run>/attempt-N``): same run id, record, plan and watcher; each attempt has its own ledger row.
``ledger`` rows are the old spend.json rows (block "" = top-level "runs", else e.g. "nebius" or
"posttrain_pilot_v1"); ``queue`` is the run's live/queued row. ``order`` keeps the old files' order; new entries
use epoch milliseconds, so they sort after every migrated one.

Launch: ``reserve(...)`` checks the gate (the arm's own allowance, then the global caps) and writes the ledger and
live-queue rows under one short lock. End: ``settle(...)``. Edits: ``update(run, fn)``.
"""
import contextlib
import fcntl
import json
import os
import re
import sys
import time
from pathlib import Path


def _default_root():
    """The main checkout, also from an arm worktree (.claude/worktrees/<name>): one set of records for every agent."""
    here = Path(__file__).resolve().parents[3]
    return here.parents[2] if here.parent.name == "worktrees" and here.parent.parent.name == ".claude" else here


ROOT = Path(os.environ.get("ARC_RECORDS_ROOT") or _default_root())


def _root(root=None):
    """``root`` (default ROOT), with an arm worktree mapped to the main checkout: records live in main only."""
    p = Path(root or ROOT)
    return p.parents[2] if p.parent.name == "worktrees" and p.parent.parent.name == ".claude" else p
END_STATES = {"complete", "completed", "done", "finished", "stopped", "failed", "cancelled", "canceled", "ended",
              "settled", "observed_lifetime", "provider_metered", "metered", "reconciled", "lost",
              "launch_failed_no_execution", "stopped_by_user"}


END_CAUSES = ("completed", "stopped-owner", "preempt", "capacity", "superseded")  # plus "bug:<class>"
STATES = ("reserved", "running", "ended")


class GateError(RuntimeError):
    pass


def check_cause(cause, allow_unclassified=False):
    """``cause`` (lower-cased) if valid: completed | stopped-owner | preempt | capacity | superseded | bug:<class>."""
    cause = cause.strip().lower() if isinstance(cause, str) else cause
    if cause in END_CAUSES or (allow_unclassified and cause == "unclassified") or re.fullmatch(
            r"bug:[a-z0-9][a-z0-9._-]*", str(cause or "")):
        return cause
    raise ValueError(f"end_cause {cause!r}: one of {', '.join(END_CAUSES)} or bug:<class> (e.g. bug:nccl-hang)")


def state_of(rec):
    """reserved | running | ended (records from before 24 Oct carry no ``state``: derived from the queue/ledger)."""
    if rec.get("state") in STATES:
        return rec["state"]
    q = rec.get("queue")
    if q and q["section"] == "live":
        return "running"
    led = rec.get("ledger") or []
    return "reserved" if led and led[-1]["row"].get("state") in OPEN_STATES else "ended" if led else "reserved"


def attempt_id(run, n):
    return run if n <= 1 else f"{run}/attempt-{n}"


def runs_dir(root=None):
    return _root(root) / "docs" / "runs"


def budget_path(root=None):
    return _root(root) / "docs" / "budget.json"


def fname(run):
    return re.sub(r"[^A-Za-z0-9._+-]", "_", run)[:200] + ".json"


def _now_order():
    return int(time.time() * 1000)


@contextlib.contextmanager
def _locked(path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path.with_name(path.name + ".lock"), "a") as lk:
        fcntl.flock(lk, fcntl.LOCK_EX)
        yield


def _write(path, data):
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    tmp.write_text(json.dumps(data, indent=1, ensure_ascii=False) + "\n")
    os.replace(tmp, path)


def get(run, root=None):
    p = runs_dir(root) / fname(run)
    return json.loads(p.read_text()) if p.is_file() else None


def update(run, fn, root=None):
    """Read-modify-write one record under its own lock; ``fn(rec)`` mutates it. Returns the record."""
    p = runs_dir(root) / fname(run)
    with _locked(p):
        rec = json.loads(p.read_text()) if p.is_file() else {"run": run}
        if rec.get("run") != run:
            raise ValueError(f"{p.name} holds {rec.get('run')!r}, not {run!r} (sanitized name clash)")
        fn(rec)
        _write(p, rec)
    return rec


def records(root=None):
    d = runs_dir(root)
    return [json.loads(p.read_text()) for p in sorted(d.glob("*.json"))] if d.is_dir() else []


def billed_path(root=None):  # Modal's bill to date: a generated, untracked view (dash loop, hourly)
    return _root(root) / "docs" / "modal-billed.json"


def load_budget(root=None):
    b = json.loads(budget_path(root).read_text())
    try:
        b["modal_billed"] = json.loads(billed_path(root).read_text())
    except (OSError, ValueError):
        pass
    return b


# --- views ---------------------------------------------------------------------------------------------------------
def build(root=None, recs=None, budget=None):
    """(queue, spend): the legacy files' content generated from the records and docs/budget.json."""
    recs = records(root) if recs is None else recs
    budget = load_budget(root) if budget is None else budget
    blocks = {}
    for r in recs:
        led = r.get("ledger") or []
        for i, e in enumerate(led):
            row = e["row"]
            if i == len(led) - 1 and r.get("state") == "running" and row.get("state") == "reserved":
                row = {**row, "state": "running"}  # the record's lifecycle; the row's own state stays the launcher's
            blocks.setdefault(e["block"], []).append((e["order"], row))
    spend = {}
    for key in budget["spend_order"]:
        if key == "runs":
            spend["runs"] = [row for _, row in sorted(blocks.get("", []), key=lambda x: x[0])]
        elif key in budget["blocks"]:
            meta = budget["blocks"][key]
            spend[key] = {k: ([row for _, row in sorted(blocks.get(key, []), key=lambda x: x[0])] if k == "runs"
                              else meta[k]) for k in meta["_order"]}
        else:
            spend[key] = budget["spend"][key]
    sections = {"queued": [], "live": []}
    desc, cpu = [], []
    for r in recs:
        q = r.get("queue")
        if q:
            sections.setdefault(q["section"], []).append((q["order"], q["row"]))
        if r.get("description"):
            desc.append((r["description"]["order"], r["run"], r["description"]["text"]))
        if r.get("cpu_app"):
            cpu.append((r["cpu_app"]["order"], r["run"]))
    qb = budget["queue"]
    queue = {}
    for key in qb["_order"]:
        if key == "descriptions":
            queue[key] = {run: text for _, run, text in sorted(desc)}
        elif key == "cpu_apps":
            queue[key] = [run for _, run in sorted(cpu)]
        elif key in sections:
            queue[key] = [row for _, row in sorted(sections[key], key=lambda x: x[0])]
        else:
            queue[key] = qb[key]
    return queue, spend


def write_views(root=None):
    """Regenerate docs/modal-queue.json and docs/spend.json (indent=1) from the records, in place under the same
    flock the legacy writers take (a replaced inode would swallow a waiting writer's edit)."""
    root = _root(root)
    with _legacy_locked(root) as files:
        _write_views_locked(root, files)


@contextlib.contextmanager
def _legacy_locked(root):
    files = {}
    try:
        for name in ("modal-queue.json", "spend.json"):
            path = root / "docs" / name
            f = open(path, "r+" if path.exists() else "w+")  # untracked: absent in a fresh checkout
            fcntl.flock(f, fcntl.LOCK_EX)
            files[name] = f
        yield files
    finally:
        for f in files.values():
            f.close()


def _write_views_locked(root, files):
    queue, spend = build(root)
    for name, data in (("modal-queue.json", queue), ("spend.json", spend)):
        f = files[name]
        text = json.dumps(data, indent=1, ensure_ascii=False) + "\n"
        f.seek(0)
        if f.read() != text:
            f.seek(0)
            f.write(text)
            f.truncate()
            f.flush()


# --- migration (one-off at cut-over; also the sync of legacy edits during it) -----------------------------------------
def split(queue, spend, arm_rules=None):
    """{run: record} and the budget from the legacy files' content."""
    recs = {}

    def rec(run):
        return recs.setdefault(run, {"run": run})

    budget = {"spend_order": list(spend), "spend": {}, "blocks": {}, "queue": {"_order": list(queue)}}
    for key, val in spend.items():
        rows = val if key == "runs" else val.get("runs") if isinstance(val, dict) and isinstance(val.get("runs"), list) \
            else None
        if rows is None:
            budget["spend"][key] = val
            continue
        block = "" if key == "runs" else key
        if block:
            budget["blocks"][key] = {"_order": list(val), **{k: v for k, v in val.items() if k != "runs"}}
        for i, row in enumerate(rows):
            rec(str(row.get("run_id") or row.get("run"))).setdefault("ledger", []).append(
                {"block": block, "order": i, "row": row})
    for key, val in queue.items():
        if key == "descriptions":
            for i, (run, text) in enumerate(val.items()):
                rec(run)["description"] = {"order": i, "text": text}
        elif key == "cpu_apps":
            for i, app in enumerate(val):
                rec(app)["cpu_app"] = {"order": i}
        elif key in ("live", "queued"):
            for i, row in enumerate(val):
                rec(row["run"])["queue"] = {"section": key, "order": i, "row": row}
        else:
            budget["queue"][key] = val
    for run, r in recs.items():
        arm = (r.get("queue") or {}).get("row", {}).get("arm") or next(
            (e["row"].get("arm") for e in r.get("ledger") or [] if e["row"].get("arm")), None)
        if not arm and arm_rules:
            arm = arm_for(run, arm_rules)
        if arm:
            r["arm"] = arm
    return recs, budget


def arm_rules(root=None):
    """Run-id prefix -> arm (docs/html/arm-state.json "run_arms")."""
    try:
        return json.loads((_root(root) / "docs/html/arm-state.json").read_text()).get("run_arms") or {}
    except (OSError, ValueError):
        return {}


def arm_for(run, rules):
    """The arm of the longest matching prefix ("b2-mfu" beats "b2-"), or None."""
    hits = [k for k in rules if run.lower().startswith(k.lower())]
    return rules[max(hits, key=len)] if hits else None


def migrate(root=None, arm_rules=None):
    """Write docs/runs/*.json and the views' layout into docs/budget.json from the current legacy files."""
    root = _root(root)
    queue = json.loads((root / "docs/modal-queue.json").read_text())
    spend = json.loads((root / "docs/spend.json").read_text())
    recs, layout = split(queue, spend, arm_rules)
    names = {}
    for run in recs:
        names.setdefault(fname(run), []).append(run)
    clash = {k: v for k, v in names.items() if len(v) > 1}
    if clash:
        raise ValueError(f"sanitized file-name clashes: {clash}")
    d = runs_dir(root)
    d.mkdir(parents=True, exist_ok=True)
    for run, r in recs.items():
        _write(d / fname(run), r)
    bp = budget_path(root)
    old = json.loads(bp.read_text()) if bp.is_file() else {}
    _write(bp, {**old, **layout})
    state = root / "runs/ops/records"  # sync's baseline: without one, the first sync could not tell who changed what
    state.mkdir(parents=True, exist_ok=True)
    q, s = build(root)
    _write(state / "last-queue.json", q)
    _write(state / "last-spend.json", s)
    return len(recs)


# --- the gate and the writes ------------------------------------------------------------------------------------------
GPU_KINDS = {"B300": r"B300", "B200": r"B200", "RTX PRO 6000": r"RTX[ -]?PRO[ -]?6000", "L4": r"\bL4\b",
             "H200": r"H200", "H100": r"H100", "A100": r"A100", "L40S": r"L40S"}  # all Modal kinds: the 10-GPU cap


def gpus_of(row):
    """{kind: count} of a queue/ledger row ("2 x B300", "RTX PRO 6000", gpu_count)."""
    text = str(row.get("gpu") or "")
    out = {}
    for kind, pat in GPU_KINDS.items():
        if re.search(pat, text, re.I):
            m = re.search(rf"(\d+)\s*[x×]\s*{pat}|{pat}\s*[:x×]\s*(\d+)", text, re.I)
            n = int(next(g for g in m.groups() if g)) if m else int(row.get("gpu_count") or 1)
            out[kind] = n
    return out


def live_gpus(recs, arm=None, exempt_long=False, arms=None):
    """{kind: count} over live queue rows (of ``arm`` or the ``arms`` set; an arm's count leaves out its rows
    launched outside its allowance, e.g. throughput's c100 300K quality reads)."""
    tot = {}
    for r in recs:
        q = r.get("queue")
        if not q or q["section"] != "live" or (arm and r.get("arm") != arm) or (arms and r.get("arm") not in arms):
            continue
        if exempt_long and q["row"].get("long_run_24h"):
            continue
        led = (r.get("ledger") or [{}])[-1].get("row") or {}
        if arm and (q["row"].get("allowance_exempt") or led.get("allowance_exempt")):  # queue rows reconcile adds
            continue                                                                    # carry no flag
        if exempt_long and led.get("long_run_24h"):
            continue
        for k, n in gpus_of(q["row"]).items():
            tot[k] = tot.get(k, 0) + n
    return tot


MODAL_GPU_RATE = {"B300": 0.001972, "B200": 0.001736, "H200": 0.001261}  # USD per GPU-second (Modal list price)
OPEN_STATES = {"reserved", "running", "submitted", "launched", "pending", "queued"}
ACTUAL_KEYS = ("actual_usd", "cost_usd", "provider_gross_usd", "gross_metered_usd", "charged_usd",
               "charged_usd_estimate", "measured_usd_estimate")


def row_cost(row):
    """(usd, kind): an open reservation counts at its bound; a settled row at its actual cost (a billed or measured
    figure, or the bound the settle step reset to observed seconds: previous_charged_bound_usd present), else its
    GPU seconds x rate ("derived"), else its bound, flagged "no-actual" (never the all-time bound
    when an actual exists)."""
    bound = float(row.get("charged_bound_usd") or 0)
    if row.get("state") in OPEN_STATES:
        return bound, "open"
    for k in ACTUAL_KEYS:
        if isinstance(row.get(k), (int, float)):
            return float(row[k]), "actual"
    if "previous_charged_bound_usd" in row:
        return bound, "actual"
    kind = next(iter(gpus_of(row)), None)
    per_gpu = float(row.get("gpu_rate_per_second") or 0) or (MODAL_GPU_RATE.get(kind, 0.0)
                                                            if row.get("platform", "modal") == "modal" else 0.0)
    rate = per_gpu * int(row.get("gpu_count") or 1) or float(row.get("rate_usd_per_hour") or 0) / 3600
    secs = row.get("observed_seconds") or row.get("measured_app_seconds") or row.get("elapsed_seconds")
    if not secs and row.get("start_epoch") and row.get("end_epoch"):
        secs = float(row["end_epoch"]) - float(row["start_epoch"])
    if rate and secs and float(secs) > 0:
        return min(bound, float(secs) * rate), "derived"  # GPU time at list rate (no CPU/RAM share): a floor
    return bound, "no-actual"


def committed_usd(recs, arm=None, block=None, since=0.0):
    """Open reservations at their bound plus settled rows at their actual cost (row_cost)."""
    s = 0.0
    for r in recs:
        if arm and r.get("arm") != arm:
            continue
        for e in r.get("ledger") or []:
            if block is not None and e["block"] != block:
                continue
            if since and float(e["row"].get("start_epoch") or e["order"] / 1000 or 0) < since:
                continue
            s += row_cost(e["row"])[0]
    return s


def gate(arm, gpu, gpu_count, bound_usd, *, platform, long_run_24h=False, allowance_exempt=None, root=None,
         recs=None, budget=None, modal=False):
    """Raises GateError unless the launch fits the ARM'S OWN allowance (its GPUs by kind, or the shared pool it
    belongs to, e.g. the post-training B300 pool; its dollars) and the global caps (GPUs by kind, platform dollars).
    Other arms' allowances are not checked. ``allowance_exempt`` (a reason, owner-approved): skips the arm's GPU
    allowance only (e.g. throughput's c100 300K quality reads); the global caps still apply. An allowance with
    "usd_mode": "warn" reports its dollar overage instead of refusing. Returns the warnings (list of str)."""
    warnings = []
    recs = records(root) if recs is None else recs
    budget = load_budget(root) if budget is None else budget
    want = gpus_of({"gpu": gpu, "gpu_count": gpu_count})
    allow = (budget.get("allowances") or {}).get(arm)
    if allow is None:
        raise GateError(f"arm {arm!r} has no allowance in docs/budget.json")
    if not allowance_exempt:
        mine = live_gpus(recs, arm)
        for kind, n in want.items():
            pools = [(name, p) for name, p in (budget.get("pools") or {}).items()
                     if p["kind"] == kind and arm in p["arms"]]
            cap = (allow.get("gpus") or {}).get(kind)
            if cap is None and not pools:
                cap = 0
            if cap is not None and mine.get(kind, 0) + n > cap:
                raise GateError(f"{arm}: {mine.get(kind, 0)} {kind} live + {n} > its allowance {cap}")
            for name, p in pools:
                used = live_gpus(recs, arms=set(p["arms"])).get(kind, 0)
                if used + n > p["cap"]:
                    raise GateError(f"{arm}: pool {name} has {used} {kind} live + {n} > {p['cap']}")
    if allow.get("usd") is not None:
        used = committed_usd(recs, arm, since=float(allow.get("usd_since_epoch") or 0))
        if used + bound_usd > allow["usd"]:
            msg = f"{arm}: ${used:.2f} committed + ${bound_usd:.2f} > its ${allow['usd']:.0f} allowance"
            if allow.get("usd_mode") != "warn":
                raise GateError(msg)
            warnings.append(msg + " (warn-only)")
    if platform == "modal" and want and arm not in MODAL_RESERVED_ARMS:
        check_modal_slots(arm, sum(want.values()), recs, modal_live(root) if modal is False else modal)
    caps = budget.get("caps") or {}
    every = live_gpus(recs, exempt_long=long_run_24h)
    for kind, n in want.items():
        cap = (caps.get("gpus") or {}).get(kind)
        if cap is not None and not (long_run_24h and kind == "RTX PRO 6000") and every.get(kind, 0) + n > cap:
            raise GateError(f"global: {every.get(kind, 0)} {kind} live + {n} > cap {cap}")
    pcap = (caps.get("usd") or {}).get(platform)
    if pcap is not None:
        block = "nebius" if platform == "nebius" else ""
        billed = budget.get("modal_billed") if platform == "modal" else None
        if billed:  # Modal's own bill to date (records.py billing) + open reservations at their bound
            used = float(billed["usd"]) + sum(float(e["row"].get("charged_bound_usd") or 0) for r in recs
                                              for e in r.get("ledger") or [] if e["block"] == ""
                                              and e["row"].get("state") in OPEN_STATES)
        else:
            used = committed_usd(recs, block=block) + (float(budget["spend"].get("infrastructure_reserve_usd") or 0)
                                                       if block == "" else 0)
        if used + bound_usd > pcap:
            raise GateError(f"global {platform}: ${used:.2f} committed + ${bound_usd:.2f} > cap ${pcap:.0f}")
    return warnings


MODAL_GPU_CAP, MODAL_RESERVED, MODAL_RESERVED_ARMS = 10, 6, ("rl", "sft")  # Modal's 10 GPUs; RL+PTH keep 6


def modal_live(root=None):
    """{app_id: {"name", "gpus": {kind: n}}} of live Modal containers (tools/ops/modal_gpus.py, Modal itself); None if
    Modal cannot be asked (the gate then counts the live Modal queue rows)."""
    import subprocess

    root = _root(root)
    py = root / ".venv/bin/python"
    try:
        r = subprocess.run([str(py if py.exists() else sys.executable), str(root / "tools/ops/modal_gpus.py")],
                           capture_output=True, text=True, timeout=120)
        data = json.loads(r.stdout or "{}")
        return None if r.returncode or "error" in data else data
    except (OSError, ValueError, subprocess.SubprocessError):
        return None


def modal_slots(recs, live=None):
    """(total, reserved-arm, other) live Modal GPUs. ``live``: modal_live() (apps attributed to arms through the app_id
    on their records; an unknown app counts as other); None: the live Modal queue rows of the records."""
    if live is None:
        rows = [r for r in recs if (r.get("queue") or {}).get("section") == "live"
                and str(r["queue"]["row"].get("platform") or "").lower() == "modal"]
        res = sum(sum(gpus_of(r["queue"]["row"]).values()) for r in rows if r.get("arm") in MODAL_RESERVED_ARMS)
        other = sum(sum(gpus_of(r["queue"]["row"]).values()) for r in rows if r.get("arm") not in MODAL_RESERVED_ARMS)
        return res + other, res, other
    arm_of = {}
    for r in recs:
        for row in [(r.get("queue") or {}).get("row") or {}] + [e["row"] for e in r.get("ledger") or []]:
            for key in ("app_id", "modal_app_id"):
                if str(row.get(key) or "").startswith("ap-"):
                    arm_of[row[key]] = r.get("arm")
    res = other = 0
    for app, info in live.items():
        n = sum(int(v) for v in (info.get("gpus") or {}).values())
        if arm_of.get(app) in MODAL_RESERVED_ARMS:
            res += n
        else:
            other += n
    return res + other, res, other


def modal_others_limit(reserved_live):
    """GPUs every non-RL/PTH Modal job may hold together: 4, never cutting into RL+PTH's 6 (less if RL+PTH hold more)."""
    return MODAL_GPU_CAP - max(reserved_live, MODAL_RESERVED)


def check_modal_slots(arm, n, recs, live=None):
    """a non-RL/PTH Modal GPU launch is refused when the other arms' live Modal GPUs (all types) + n
    would exceed 10 - max(6, RL/PTH live)."""
    if arm in MODAL_RESERVED_ARMS or n <= 0:
        return
    total, res, other = modal_slots(recs, live)
    if other + n > modal_others_limit(res):
        raise GateError(f"modal slots: {other} other-arm GPUs live on Modal ({res} RL/PTH, {total} total) + {n} > "
                        f"{modal_others_limit(res)} "
                        f"(Modal's cap is {MODAL_GPU_CAP}; RL+PTH keep {MODAL_RESERVED}): wait or run on Nebius")


@contextlib.contextmanager
def _gate_lock(root=None):
    with _locked(runs_dir(root) / ".gate"):
        yield


def reserve(run, *, arm, platform, gpu, gpu_count=1, bound_usd, max_seconds, authorization, what, kind=None,
            block=None, long_run_24h=False, allowance_exempt=None, root=None, **row):
    """Gate, then the ledger row (state reserved) and the live queue row, atomically against other launches. A run
    whose record has ended gets attempt N+1 (``<run>/attempt-N``) in the same record; an open one is refused."""
    block = ("nebius" if platform == "nebius" else "") if block is None else block
    with _gate_lock(root):
        old = get(run, root)
        n = 1
        if old and old.get("ledger"):
            if state_of(old) != "ended":
                raise GateError(f"{run}: attempt {old.get('attempt', 1)} is still {state_of(old)}; settle it first "
                                f"(records.py settle {run} STATE --cause ...)")
            n = int(old.get("attempt") or 1) + 1
        warnings = gate(arm, gpu, gpu_count, bound_usd, platform=platform, long_run_24h=long_run_24h,
                        allowance_exempt=allowance_exempt, root=root)
        for w in warnings:
            print(f"GATE WARNING: {w}", file=sys.stderr)
        now = time.time()
        order = _now_order()

        def fill(rec):
            rec["arm"] = arm
            if n > 1 and not rec.get("attempts"):  # a pre-attempts record: its history becomes attempt 1
                rec["attempts"] = [{"n": 1, "state": "ended", "end_cause": rec.get("end_cause") or "unclassified",
                                    "start_epoch": rec["ledger"][-1]["row"].get("start_epoch"),
                                    "end_epoch": rec["ledger"][-1]["row"].get("end_epoch")}]
            rec.update(state="reserved", attempt=n)
            rec.pop("end_cause", None)
            rec.setdefault("attempts", []).append({"n": n, "state": "reserved", "start_epoch": round(now)})
            rec.setdefault("ledger", []).append({"block": block, "order": order, "row": {
                "run_id": attempt_id(run, n), "arm": arm, "state": "reserved", "attempt": n, "kind": kind, "platform": platform, "gpu": gpu,
                "gpu_count": gpu_count, "charged_bound_usd": round(float(bound_usd), 4), "max_seconds": max_seconds,
                "start_epoch": round(now), "authorization": authorization,
                **({"gpu_rate_per_second": MODAL_GPU_RATE[k]} if platform == "modal" and (k := next(
                    iter(gpus_of({"gpu": gpu})), None)) in MODAL_GPU_RATE else {}), **row}})
            rec["queue"] = {"section": "live", "order": order, "row": {
                "run": run, "arm": arm, "platform": platform, "gpu": gpu, "gpu_count": gpu_count, "state": "running",
                "start": time.strftime("%H:%M", time.gmtime(now)), "start_epoch": round(now), "what": what,
                **({"attempt": n} if n > 1 else {}),
                **({"long_run_24h": True} if long_run_24h else {}),
                **({"allowance_exempt": allowance_exempt} if allowance_exempt else {})}}

        rec = update(run, fill, root)
    publish(root)
    return rec


def require_reservation(run, root=None, max_age_s=6 * 3600, modal=False):
    """Raises GateError unless ``run`` holds an open reservation (state "reserved": records.reserve passed the gate)
    made within ``max_age_s``. Every launcher calls it right before its cloud call (modal run / spawn, VM create), so
    a refused or skipped reserve can never launch (24 Oct: a launch went ahead after a printed refusal)."""
    rec = get(run, root)
    if not rec or not rec.get("ledger"):
        raise GateError(f"{run}: no reservation (records.py reserve first; a refused reserve blocks the launch)")
    st = state_of(rec)
    if st != "reserved":
        raise GateError(f"{run}: attempt {rec.get('attempt', 1)} is {st}, not reserved (reserve a new attempt after a "
                        f"settle; never launch twice on one reservation)")
    t0 = float(rec["ledger"][-1]["row"].get("start_epoch") or 0)
    if time.time() - t0 > max_age_s:
        raise GateError(f"{run}: reservation is {(time.time() - t0) / 3600:.1f} h old (> {max_age_s / 3600:.0f} h): "
                        f"settle it and reserve again")
    row = (rec.get("queue") or {}).get("row") or rec["ledger"][-1]["row"]
    if str(row.get("platform") or rec["ledger"][-1]["row"].get("platform") or "").lower() == "modal":
        others = [r for r in records(root) if r.get("run") != run]  # its own row is not up yet
        check_modal_slots(rec.get("arm"), sum(gpus_of(row).values()), others, modal_live(root) if modal is False else modal)
    return rec


def publish(root=None):
    """After a record write: regenerate the views (skipped with ARC_RECORDS_NO_SYNC=1)."""
    if os.environ.get("ARC_RECORDS_NO_SYNC") != "1":
        sync(root)


def set_queue(run, root=None, section="live", **fields):
    def fn(rec):
        if state_of(rec) == "ended" and not rec.get("queue"):  # settled: a late what/next edit must not re-open a bare live row
            return
        q = rec.get("queue") or {"section": section, "order": _now_order(), "row": {"run": run}}
        q["row"].update(fields)
        rec["queue"] = q

    rec = update(run, fn, root)
    publish(root)
    return rec


def _mark_running(rec):
    if state_of(rec) != "ended" and rec.get("state") != "running":
        rec["state"] = "running"
        if rec.get("attempts"):
            rec["attempts"][-1]["state"] = "running"


def set_ledger(run, root=None, **fields):
    """Patch the run's latest ledger row (e.g. app_id, charged_bound_usd) without ending it. A real app id
    (``ap-...``) or ``running=True`` marks the run running."""
    go = bool(fields.pop("running", False)) or str(fields.get("app_id") or "").startswith("ap-")

    def fn(rec):
        if not rec.get("ledger"):
            raise GateError(f"{run}: no ledger row (records.py reserve first)")
        rec["ledger"][-1]["row"].update(fields)
        if go:
            _mark_running(rec)

    rec = update(run, fn, root)
    publish(root)
    return rec


def add_result(run, metrics, *, bundle=None, vs=None, verdict=None, note=None, t=None, by=None, root=None):
    """Append one read of the run (results live in the run record, not scattered in arm logs):
    ``metrics`` {name: number}, e.g. {"score300k": 13.7, "tok_s": 1534}; ``bundle`` the code/bundle id; ``vs`` the
    run(s) it is compared with; ``verdict`` one line. A later read of the same run is appended, never merged."""
    entry = {"t": round(float(t if t is not None else time.time())), "metrics": dict(metrics),
             **{k: v for k, v in (("bundle", bundle), ("vs", vs), ("verdict", verdict), ("note", note), ("by", by))
                if v is not None}}

    def fn(rec):
        rec.setdefault("results", []).append(entry)

    return update(run, fn, root)


def results(*, arm=None, since=0.0, metric=None, prefix=None, root=None, recs=None):
    """[(run, arm, entry)] sorted by time: every read since ``since`` of runs of ``arm`` / with run-id ``prefix``,
    that report ``metric`` (if given)."""
    out = []
    for r in records(root) if recs is None else recs:
        if (arm and r.get("arm") != arm) or (prefix and not r["run"].startswith(prefix)):
            continue
        for e in r.get("results") or []:
            if e["t"] >= since and (metric is None or metric in e.get("metrics", {})):
                out.append((r["run"], r.get("arm"), e))
    return sorted(out, key=lambda x: x[2]["t"])


def settle(run, state, *, end_cause=None, root=None, **fields):
    """End a run's current attempt: the latest ledger row gets ``state`` (the billing state: completed, stopped,
    failed ...) and ``end_cause`` (required: why it ended, check_cause) plus fields (observed_seconds, end_epoch,
    note); the record is ended and its queue row removed. Settling an ended record again corrects its cause."""
    end_cause = check_cause(end_cause)
    end = round(fields.pop("end_epoch", time.time()))

    def fn(rec):
        if not rec.get("ledger") and not rec.get("queue"):
            raise GateError(f"{run}: no such run record (typo? records.py show {run})")
        if rec.get("ledger"):
            rec["ledger"][-1]["row"].update(state=state, end_cause=end_cause, end_epoch=end, **fields)
        rec.update(state="ended", end_cause=end_cause)
        if rec.get("attempts"):
            rec["attempts"][-1].update(state="ended", end_cause=end_cause, end_epoch=end)
        rec.pop("queue", None)

    rec = update(run, fn, root)
    publish(root)
    return rec


def unclassified(root=None, recs=None):
    """Runs ended without a stated cause (app/VM gone, no settle): their arms must settle them with --cause."""
    return [r["run"] for r in (records(root) if recs is None else recs)
            if r.get("ledger") and r.get("end_cause") == "unclassified"]


def backfill_states(root=None, causes=None):
    """One-off: give every record its ``state`` (state_of) and, for ended ones, ``end_cause`` from ``causes``
    {run: cause} or "completed" for a completed/finished ledger row, else "unclassified". Returns the count changed."""
    causes = causes or {}
    n = 0
    for r in records(root):
        if not r.get("ledger"):
            continue
        want_state = state_of(r)
        led = (r.get("ledger") or [{}])[-1].get("row") or {}
        cause = causes.get(r["run"]) or r.get("end_cause") or (
            "completed" if str(led.get("state")).lower() in ("complete", "completed", "done", "finished") else
            "stopped-owner" if led.get("state") == "stopped_by_user" else "unclassified")
        if want_state != "ended":
            cause = None
        if r.get("state") == want_state and r.get("end_cause") == cause:
            continue
        check_cause(cause, allow_unclassified=True) if cause else None

        def fn(rec, st=want_state, c=cause):
            rec["state"] = st
            if c:
                rec["end_cause"] = c
                if rec.get("ledger"):
                    rec["ledger"][-1]["row"]["end_cause"] = c
        update(r["run"], fn, root)
        n += 1
    return n


def register_cpu_app(app_id, name=None, root=None):
    def fn(rec):
        rec.setdefault("cpu_app", {"order": _now_order()})
        if name:
            rec["cpu_app"]["name"] = name

    rec = update(app_id, fn, root)
    publish(root)
    return rec


def drop_cpu_app(app_id, root=None):
    """Remove a CPU-app registration (a mistyped or finished app id). Refuses a record that holds anything else."""
    p = runs_dir(root) / fname(app_id)
    with _locked(p):
        rec = json.loads(p.read_text()) if p.is_file() else None
        if rec is None:
            raise GateError(f"{app_id}: no such record")
        extra = sorted(k for k in rec if k not in ("run", "cpu_app"))
        if extra:
            raise GateError(f"{app_id}: not a CPU-app-only record (has {', '.join(extra)})")
        p.unlink()
    publish(root)


def autoregister_cpu_apps(root=None, live=None):
    """Live Modal apps with no GPU and no queue row/CPU registration -> registered as CPU apps named "auto:<app>"
    (CPU jobs pass no gate). Returns [(app_id, name)]. GPU apps without a row are left to gpu_reconcile's note."""
    live = modal_live(root) if live is None else live
    if not live:
        return []
    known = set()
    for r in records(root):
        known.add(r.get("run"))
        row = (r.get("queue") or {}).get("row") or {}
        if row.get("app_id"):
            known.add(row["app_id"])
    out = []
    for aid, app in sorted(live.items()):
        if aid in known or any((app.get("gpus") or {}).values()):
            continue
        name = f"auto:{app.get('name') or '?'}"
        register_cpu_app(aid, name=name, root=root)
        out.append((aid, name))
    return out


# --- views refresh (phase D: legacy view edits are no longer imported) -------------------------------------------------
PARTS = ("ledger", "queue", "description", "cpu_app")


def sync(root=None, state_dir=None, arm_rules=None):
    """One dash pass: regenerate the views from the records. A run a direct writer changed in a view since the last
    generated copy is not imported (records.py is the only writer since phase D, 24 Oct): it is appended to
    <state>/dropped-legacy.jsonl for the dashboard banner. The last-*.json copies stay current so an arm branch still
    on the cut-over code imports nothing stale. Returns the dropped run ids."""
    root = _root(root)
    state_dir = Path(state_dir or root / "runs/ops/records")
    state_dir.mkdir(parents=True, exist_ok=True)
    with _gate_lock(root), _legacy_locked(root) as files:
        cur_q, cur_s = (json.loads(files[n].read() or "{}") for n in ("modal-queue.json", "spend.json"))
        last_q, last_s = state_dir / "last-queue.json", state_dir / "last-spend.json"
        dropped = []
        if cur_q and cur_s and last_q.is_file() and last_s.is_file():
            now_recs, _ = split(cur_q, cur_s, arm_rules)
            old_recs, _ = split(json.loads(last_q.read_text()), json.loads(last_s.read_text()), arm_rules)
            gen_recs, _ = split(*build(root), arm_rules)  # a view another writer regenerated is not an edit

            def differs(a, b, run):
                return any(_rows(a.get(run, {}).get(k)) != _rows(b.get(run, {}).get(k)) for k in PARTS)
            dropped = sorted(run for run in set(now_recs) | set(old_recs) | set(gen_recs)
                             if differs(now_recs, old_recs, run) and differs(now_recs, gen_recs, run))
        if dropped:
            with open(state_dir / "dropped-legacy.jsonl", "a") as f:
                f.write(json.dumps({"t": time.time(), "runs": dropped}) + "\n")
        _write_views_locked(root, files)
        q, s = build(root)
        _write(last_q, q)
        _write(last_s, s)
    return dropped


def _rows(part):
    """A record part without the generated sort keys (split() numbers rows by file position)."""
    if isinstance(part, list):
        return [e.get("row") for e in part]
    return part.get("row", part.get("text")) if isinstance(part, dict) else part


def dropped_legacy(root=None, since=0.0):
    """Runs whose direct view edits were dropped since ``since`` (epoch)."""
    p = _root(root) / "runs/ops/records/dropped-legacy.jsonl"
    out = set()
    for line in (p.read_text().splitlines() if p.is_file() else []):
        e = json.loads(line)
        if e["t"] >= since:
            out.update(e["runs"])
    return sorted(out)


def snapshot(root=None):
    """Ended records -> docs/runs-settled/<run>.json (tracked in git; written only when changed). Returns the count."""
    d = _root(root) / "docs" / "runs-settled"
    d.mkdir(parents=True, exist_ok=True)
    n = 0
    for rec in records(root):
        if state_of(rec) != "ended":
            continue
        p = d / fname(rec["run"])
        text = json.dumps(rec, indent=1, ensure_ascii=False) + "\n"
        if not p.is_file() or p.read_text() != text:
            p.write_text(text)
            n += 1
    return n


def apply_queue_ops(ops, root=None):
    """reconcile.py's queue edits on the records, then one publish: ("set", run, fields) live-row fields; ("move",
    run, fields) queued -> live; ("add", run, row) a live row unless the run has one; ("drop", run, _) remove it."""
    for op, run, f in ops:
        def fn(rec, op=op, f=f):
            q = rec.get("queue")
            if op == "set" and q and q["section"] == "live":
                q["row"].update(f)
                _mark_running(rec)  # its app/VM is listed live
            elif op == "move" and q and q["section"] == "queued":
                q.update(section="live", order=_now_order())
                q["row"].update(f)
                _mark_running(rec)
            elif op == "add" and not q:
                rec["queue"] = {"section": "live", "order": _now_order(), "row": dict(f)}
                if f.get("arm") and not rec.get("arm"):
                    rec["arm"] = f["arm"]
                if rec.get("ledger"):
                    _mark_running(rec)
            elif op == "drop":
                had_live = bool(q) and q["section"] == "live"
                rec.pop("queue", None)
                if had_live and rec.get("ledger") and state_of(rec) != "ended":  # gone without a settle
                    rec.update(state="ended", end_cause="unclassified")
                    if rec.get("attempts"):
                        rec["attempts"][-1].update(state="ended", end_cause="unclassified")

        update(run, fn, root)
    if ops:
        publish(root)


def modal_billed(start="2026-09-01", modal_bin="modal"):
    """Modal's billed app cost from ``start`` to now (``modal billing report``, 31-day windows), USD."""
    import datetime
    import subprocess

    day = datetime.date.fromisoformat(start)
    today = datetime.datetime.now(datetime.UTC).date() + datetime.timedelta(days=1)
    total = 0.0
    while day < today:
        end = min(day + datetime.timedelta(days=31), today)
        out = subprocess.run([modal_bin, "billing", "report", "--start", day.isoformat(), "--end", end.isoformat(),
                              "--json"], capture_output=True, text=True, timeout=300, check=True).stdout
        total += sum(float(x["cost"]) for x in json.loads(out or "[]"))
        day = end
    return round(total, 2)


def set_modal_billed(usd, root=None):
    p = billed_path(root)
    with _locked(p):
        _write(p, {"usd": usd, "through_utc": time.strftime("%Y-%m-%d %H:%M", time.gmtime()),
                   "source": "modal billing report since 2026-09-01 (apps; volumes and egress excluded)"})
