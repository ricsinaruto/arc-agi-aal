"""One record file per run: docs/runs/<run>.json replaces the shared docs/modal-queue.json and
docs/spend.json, which become views generated from the records (``build``) so mon.py, train_dash.py,
gpu_reconcile.py and the HTML pages keep reading them. Writers touch only their run's file (no lost updates, no
merge conflicts between agents). Caps and per-arm allowances live in docs/budget.json.

Record: {"run", "arm", "state", "end_cause", "attempt", "attempts": [...], "ledger": [{"block", "order", "row"}],
         "queue": {"section", "order", "row"}, "description": {"order", "text"}, "cpu_app": {"order"},
         "plan": {...watcher plan}, "results": [...]}
``state``: reserved -> running -> ended (``state_of`` derives it for records written before 24 Oct). A relaunch of an ended run is attempt N+1 of the
same record (``<run>/attempt-N``): same run id, record, plan and watcher; each attempt has its own ledger row.
``ledger`` rows are the old spend.json rows (block "" = top-level "runs", else e.g. "nebius" or
"posttrain_pilot_v1"); ``queue`` is the run's live/queued row. ``order`` keeps the old files' order; new entries
use epoch milliseconds, so they sort after every migrated one.

Launch: ``reserve(...)`` checks the gate (the arm's own allowance, then the global caps) and writes the ledger and
live-queue rows under one short lock. Edits: ``update(run, fn)``. This copy keeps what runops.launchcfg uses; the
records CLI (settle, queue, ledger, views) is main's tools/ops/records.py.
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


ROOT = _default_root()


STATES = ("reserved", "running", "ended")


class GateError(RuntimeError):
    pass


def run_volume_name(run):
    """one Modal v2 volume per run record (attempts and preflight legs share it): arc-run-<run>, Modal's
    name rules (letters, digits, - _ ., at most 64 characters; a longer id keeps its head plus a hash). Same as main's."""
    import hashlib
    import re as _re
    base = "arc-run-" + _re.sub(r"[^A-Za-z0-9._-]", "-", run)
    return base if len(base) <= 64 else base[:55] + "-" + hashlib.sha1(run.encode()).hexdigest()[:8]


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
    return Path(root or ROOT) / "docs" / "runs"


def budget_path(root=None):
    return Path(root or ROOT) / "docs" / "budget.json"


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


def load_budget(root=None):
    return json.loads(budget_path(root).read_text())


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


@contextlib.contextmanager
def _legacy_locked(root):
    files = {}
    try:
        for name in ("modal-queue.json", "spend.json"):
            f = open(root / "docs" / name, "r+")
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
        return json.loads((Path(root or ROOT) / "docs/html/arm-state.json").read_text()).get("run_arms") or {}
    except (OSError, ValueError):
        return {}


def arm_for(run, rules):
    """The arm of the longest matching prefix ("b2-mfu" beats "b2-"), or None."""
    hits = [k for k in rules if run.lower().startswith(k.lower())]
    return rules[max(hits, key=len)] if hits else None


# --- the gate and the writes ------------------------------------------------------------------------------------------
GPU_KINDS = {"B300": r"B300", "B200": r"B200", "RTX PRO 6000": r"RTX[ -]?PRO[ -]?6000", "L4": r"\bL4\b"}


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
         recs=None, budget=None):
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


def publish(root=None):
    """After a record write: import pending legacy edits, regenerate the views."""
    sync(root)


# --- cut-over: legacy writers keep working while they migrate ----------------------------------------------------------
PARTS = ("ledger", "queue", "description", "cpu_app")


def sync(root=None, state_dir=None, arm_rules=None):
    """One pass of the cut-over sync (run by the dashboard loop): runs a legacy writer changed in
    docs/modal-queue.json / docs/spend.json since the last generated views are imported into their records (legacy
    wins for the parts it changed: ledger, queue, ...); then the views are regenerated from the records. Returns the imported run ids."""
    root = Path(root or ROOT)
    state_dir = Path(state_dir or root / "runs/ops/records")
    state_dir.mkdir(parents=True, exist_ok=True)
    with _gate_lock(root), _legacy_locked(root) as files:
        cur_q, cur_s = (json.loads(files[n].read() or "{}") for n in ("modal-queue.json", "spend.json"))
        last_q, last_s = state_dir / "last-queue.json", state_dir / "last-spend.json"
        base_q = json.loads(last_q.read_text()) if last_q.is_file() else cur_q
        base_s = json.loads(last_s.read_text()) if last_s.is_file() else cur_s
        now_recs, now_budget = split(cur_q, cur_s, arm_rules)
        old_recs, old_budget = split(base_q, base_s, arm_rules)
        changed = [run for run in set(now_recs) | set(old_recs)
                   if {k: now_recs.get(run, {}).get(k) for k in PARTS} != {k: old_recs.get(run, {}).get(k)
                                                                           for k in PARTS}]
        for run in changed:
            new, old = now_recs.get(run, {}), old_recs.get(run, {})
            parts = [p for p in PARTS if new.get(p) != old.get(p)]  # only what the legacy writer changed

            def take(rec, new=new, parts=parts):
                ended = bool(rec.get("ledger")) and state_of(rec) == "ended"
                for part in parts:
                    val = new.get(part)
                    if ended and part == "queue":  # settle is terminal: a stale view never reopens an ended run
                        continue
                    if ended and part == "ledger" and val:
                        last = rec["ledger"][-1]["row"]
                        if len(val) == len(rec["ledger"]) and val[-1]["row"].get("state") in OPEN_STATES | {None}:
                            val = val[:-1] + [{**val[-1], "row": {**val[-1]["row"], **{k: last[k] for k in (
                                "state", "end_cause", "end_epoch") if k in last}}}]
                    if val is None:
                        rec.pop(part, None)
                        continue
                    if part == "ledger":
                        mine = rec.get("ledger") or []
                        val = [{**e, "order": mine[i]["order"] if i < len(mine) else _now_order()}
                               for i, e in enumerate(val)]
                    elif part in ("queue", "description", "cpu_app"):
                        val = {**val, "order": (rec.get(part) or {}).get("order", _now_order())}
                    rec[part] = val
                if new.get("arm") and not rec.get("arm"):
                    rec["arm"] = new["arm"]
                if rec.get("ledger") and not ended and {"ledger", "queue"} & set(parts):  # lifecycle follows the edit
                    rec.pop("state", None)
                    rec["state"] = state_of(rec)
                    if rec["state"] == "ended" and not rec.get("end_cause"):
                        rec["end_cause"] = rec["ledger"][-1]["row"].get("end_cause") or "unclassified"

            update(run, take, root)
        if now_budget != old_budget:
            b = load_budget(root)
            for key in ("spend", "blocks", "queue", "spend_order"):
                b[key] = _merge_layout(b.get(key), old_budget[key], now_budget[key])
            _write(budget_path(root), b)
        _write_views_locked(root, files)
        q, s = build(root)
        _write(last_q, q)
        _write(last_s, s)
    return sorted(changed)


def _merge_layout(mine, base, theirs):
    """Legacy edits to the non-run parts (caps, histories, the queue's "next") win over an unchanged base."""
    if base == theirs:
        return mine
    if isinstance(mine, dict) and isinstance(base, dict) and isinstance(theirs, dict):
        out = dict(mine)
        for k in set(base) | set(theirs):
            if base.get(k) != theirs.get(k):
                if k in theirs:
                    out[k] = theirs[k]
                else:
                    out.pop(k, None)
        return out
    return theirs


