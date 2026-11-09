"""Reconcile docs/modal-queue.json with what actually runs, every monitor pass (mon.py). Read-only towards the clouds.
Nebius: every arc-<run id> VM that is up is a live row (made from its launch receipt if missing); a live row whose VM is gone
is dropped (its ledger row settles through `nebius_validation.py status`, as before). Modal: launch records modal-<run>.json
(docs/ and worktrees) whose ledger row is still "reserved": the call graph and the end marker on the volume decide.
Rows keep their hand-written "what"; queued rows that have not started are never dropped. Written back under flock only on change."""
import datetime, glob, json, os, pathlib, re, subprocess, sys, time
import paths
ROOT = paths.ROOT_PATH; Q = ROOT / "docs/modal-queue.json"; REC = ROOT / "docs/validation-nebius"
PY = str(ROOT / ".venv/bin/python") if (ROOT / ".venv/bin/python").exists() else sys.executable
DOWN = {"STOPPED", "STOPPING", "DELETING", "DELETED", "ERROR", "FAILED"}
GRACE = 900  # a launch this young may not be listed yet

def hm(t): return datetime.datetime.fromtimestamp(t, datetime.timezone.utc).strftime("%H:%M")

def nebius_vms():
    """{run id lowercased: {created, state}} of the arc-run VMs, or None when the listing fails (login expired, CLI error)."""
    if (paths.SCRATCH / "nebius.nologin").exists(): return None
    code = ("import json, sys; sys.path.insert(0, sys.argv[1]); import nebius_validation as nv\n"
            "print(json.dumps([[(i['metadata'].get('labels') or {}).get('arc-run') or i['metadata'].get('name', '')[4:], i['metadata'].get('created_at'),"
            " (i.get('status') or {}).get('state')] for i in nv.Cloud(nv.load_config()).instances()]))")
    rows = None
    for _ in range(3):  # transient API/CLI errors: retry before reporting the listing as failed
        try:
            r = subprocess.run([PY, "-c", code, str(paths.NEBIUS_SCRIPTS)], capture_output=True, text=True, timeout=90,
                               env={**os.environ, "ARC_NEBIUS_CONFIG": str(paths.NEBIUS_CONFIG)})
            rows = json.loads(r.stdout.strip().splitlines()[-1]) if r.returncode == 0 else None
        except Exception: rows = None
        if rows is not None: break
        time.sleep(10)
    if rows is None: return None
    out = {}
    for run, created, state in rows:
        if state in DOWN or not run: continue
        try: t = datetime.datetime.fromisoformat(created.replace("Z", "+00:00")).timestamp()
        except Exception: t = time.time()
        out[run.lower()] = {"created": t, "state": state}
    return out

def receipts():
    """{run id lowercased: path} of the Nebius launch receipts."""
    return {p.name[7:-5].lower(): p for p in REC.glob("nebius-*.json")}

def ledger_states():
    d = paths.read_json(ROOT / "docs/spend.json"); rows = list(d.get("runs") or [])
    for v in d.values():
        if isinstance(v, dict) and isinstance(v.get("runs"), list): rows += v["runs"]
    return {str(r.get("run_id") or r.get("run")): r.get("state") for r in rows if isinstance(r, dict)}

MODAL_OK = True

def modal_jobs(now):
    """{run: {start, timeout, gpu, ended}} for Modal launch records (docs/ and worktrees). A record whose ledger row is settled is
    ended; an open one (recent, row "reserved") is ended once its output holds result.json / execution.json, or when its call is
    over and no running Modal app was created at its launch (a call with retries reports failure while a new container plays)."""
    states = ledger_states(); recs = {}
    for p in glob.glob(str(ROOT / "docs/validation-*/modal-*.json")) + glob.glob(str(ROOT / ".claude/worktrees/*/docs/validation-*/modal-*.json")):
        run = pathlib.Path(p).name[6:-5]
        if run not in states: continue
        if states[run] not in ("reserved", "running", "submitted", "launched"): recs[run] = {"ended": True, "start": 0}; continue  # open-row states arms use
        try: j = json.loads(pathlib.Path(p).read_text()); mt = os.path.getmtime(p)
        except Exception: continue
        if not j.get("function_call_id"): continue
        start = j.get("launched_epoch") or j.get("started_epoch")
        if not start:  # never the file mtime: git checkouts and merges reset it, which revived 3-day-old rows as live (19 Oct)
            d = re.search(r"(20\d{6})", run)
            if not d: continue
            start = datetime.datetime.strptime(d.group(1), "%Y%m%d").replace(tzinfo=datetime.timezone.utc).timestamp()
        if now - start > (j.get("timeout_seconds") or 86400) + 3 * 3600: continue  # stale reservation
        if run not in recs or start > recs[run]["start"]:
            recs[run] = {"start": start, "timeout": j.get("timeout_seconds") or 0, "fc": j["function_call_id"], "gpu": j.get("gpu"), "output": j.get("output"), "ended": None}
    todo = {r: v for r, v in recs.items() if v["ended"] is None}
    if not todo: return recs
    code = ("import json, sys, modal\nout = {}\nfor fc in sys.argv[1:]:\n    try: out[fc] = int(modal.FunctionCall.from_id(fc).get_call_graph()[0].status)\n"
            "    except Exception as e: out[fc] = None\nprint(json.dumps(out))")
    modal = str(ROOT / ".venv/bin/modal"); ts = lambda s: datetime.datetime.fromisoformat(s).timestamp()
    try:
        r = subprocess.run([PY, "-c", code, *[v["fc"] for v in todo.values()]], capture_output=True, text=True, timeout=120)
        st = json.loads(r.stdout.strip().splitlines()[-1]) if r.returncode == 0 else {}
        apps = [(ts(a["created_at"]), a["app_id"]) for a in json.loads(subprocess.run([modal, "app", "list", "--json"], capture_output=True, text=True, timeout=60).stdout)
                if a.get("state") != "stopped"]
        boxes = {}
        for c in json.loads(subprocess.run([modal, "container", "list", "--json"], capture_output=True, text=True, timeout=60).stdout):
            boxes[c["app_id"]] = min(boxes.get(c["app_id"], 1e18), ts(c["start_time"]))
    except Exception:  # Modal listing failed this pass: carry each open run's last known state instead of dropping it
        global MODAL_OK; MODAL_OK = False
        try: prev = json.loads((paths.SCRATCH / "mon/state.json").read_text()).get("runs", {})
        except Exception: prev = {}
        for run, v in todo.items():
            p = prev.get(run)
            if p: v.update(ended=False, state=p.get("state", "running"), first=p.get("first"), since=p.get("since"))
        return recs
    def cat(vol, path):
        try: return json.loads(subprocess.run([modal, "volume", "get", vol, path, "-"], capture_output=True, text=True, timeout=60).stdout.split("✓")[0].strip() or "{}")
        except Exception: return {}
    for run, v in sorted(todo.items(), key=lambda kv: kv[1]["start"]):
        vol, _, path = (v.get("output") or f"arc-prolong-runs:/research/{run}").partition(":"); path = path.strip("/")
        try: ls = subprocess.run([modal, "volume", "ls", vol, path], capture_output=True, text=True, timeout=90).stdout
        except Exception: continue
        names = {x.rsplit("/", 1)[-1] for x in ls.split()}
        if names & {"result.json", "execution.json"}: v["ended"] = True; continue
        mine = sorted(a for a in apps if v["start"] - 1800 <= a[0] <= v["start"] + 60)  # the app its `modal run --detach` made
        app = mine[-1][1] if mine else None
        if mine: apps.remove(mine[-1])
        if not app and st.get(v["fc"]) != 0:
            if st.get(v["fc"]) is not None: v["ended"] = True
            continue
        v["ended"] = False
        started = cat(vol, f"{path}/started.json") if "started.json" in names else {}
        played = cat(vol, f"{path}/played.json") if "played.json" in names else {}
        v["first"] = min([x for x in [*(started.get("containers") or []), started.get("started_epoch")] if x] or [None]) if started else None
        if app in boxes: v["state"], v["since"] = "running", boxes[app]
        else: v["state"], v["since"] = "waiting", played.get("epoch") or v["start"]  # a call between containers: no GPU, not billed
    return recs

def no_feed_kind(run):
    """The job kind when it is one that writes no live score or tok/s (its result comes at the end), else None."""
    m = re.match(r"arc2-l4-(start|cache|drill|probe|serve)-", run)
    if m: return f"arc2-l4-{m.group(1)}"  # start/cache tests: no score feed by design
    for p in glob.glob(str(ROOT / f"docs/validation-*/modal-{run}.json")) + glob.glob(str(ROOT / f".claude/worktrees/*/docs/validation-*/modal-{run}.json")):
        try: k = json.loads(pathlib.Path(p).read_text()).get("kind") or ""
        except Exception: continue
        if k.startswith("arc2-l4-") and k not in ("arc2-l4-package", "arc2-l4-full"): return k
    return None

def ended_in_bucket(run):
    try: d = subprocess.run(["rclone", "lsf", f"nebius-arc:arc-runs-bucket/runs/validation/{run}/execution.json"], capture_output=True, text=True, timeout=120)
    except subprocess.TimeoutExpired: return False
    return d.returncode == 0 and "execution.json" in d.stdout

def arm_of(run, rules):
    hits = [k for k in rules if run.lower().startswith(k.lower())]  # longest prefix: "b2-mfu" beats "b2-"
    return rules[max(hits, key=len)] if hits else "other"

def what_of(run, arm, q, labels):
    """Text for a new row: the queue's description, else the arm log's Now line that names the run, else its label."""
    keys = [k for k in q.get("descriptions", {}) if run.startswith(k)]
    if keys: return q["descriptions"][max(keys, key=len)]
    log = ROOT / "docs/arm-logs" / f"{arm}.md"; short = re.sub(r"-(nebius|modal)-\d{8}$", "", run)
    if log.exists():
        t = log.read_text(); now = t[t.find("## Now"):t.find("## ", t.find("## Now") + 3)] if "## Now" in t else ""
        for line in re.split(r"\n(?=\s*[-*] )", now):
            if short in line: return re.sub(r"\s+", " ", re.sub(r"[`*]", "", re.sub(r"^\s*[-*]\s+", "", line))).strip()[:400]
    return next((v for k, v in labels.items() if run.startswith(k)), "")

END = ("complete", "completed", "done", "finished", "stopped", "failed", "cancelled", "canceled", "ended")


def app_ended(app_id, _c={}):
    """True when Modal no longer lists the app as running (listed once per pass; False when the listing fails)."""
    if "s" not in _c:
        try: _c["s"] = {a["app_id"] for a in ({k.lower().replace(" ", "_"): v for k, v in x.items()} for x in json.loads(subprocess.run(
                            [str(ROOT / ".venv/bin/modal"), "app", "list", "--json"], capture_output=True, text=True, timeout=60).stdout))
                        if a.get("state") != "stopped"}  # modal 1.2 vs 1.5 key styles
        except Exception: _c["s"] = None
    # only real app ids: a launcher's placeholder ("pending", None) is not an ended app
    return str(app_id or "").startswith("ap-") and _c["s"] is not None and app_id not in _c["s"]


def plan(q, vms, modal, now, st):
    """Edits as (op, run, fields): set (live row times), move (queued -> live), add, drop (from live or queued)."""
    rules, skip, labels = st.get("run_arms", {}), tuple(st.get("ignore_runs", [])), st.get("labels", {})
    rec = receipts(); ops = []
    rows = {e["run"].lower(): ("queued", e) for e in q.get("queued", [])}
    rows.update({e["run"].lower(): ("live", e) for e in q.get("live", [])})
    run_now = {}
    def full(low):  # VM labels cap at 63 characters: map a cut run id back to its receipt or queue row
        if low in rec or low in rows: return low
        hits = {k for k in (*rec, *rows) if k.startswith(low)}
        return hits.pop() if len(hits) == 1 else low
    for low, v in (vms or {}).items():
        low = full(low)
        if low.startswith(skip): continue
        path = rec.get(low); r = json.loads(path.read_text()) if path else {}
        run = r.get("run_id") or (rows.get(low) or (0, {}))[1].get("run") or low
        dur = ((r.get("spec") or {}).get("timeout") or r.get("play_seconds") or 0)
        if dur >= 86400 and "24h" not in low:  # the launcher's default 24 h VM timeout is a backstop, not the plan
            dur = (r.get("play_seconds") or 0) + 1200 if r.get("play_seconds") else 7200
        run_now[low] = {"run": run, "platform": "nebius", "gpu": "RTX PRO 6000", "start": r.get("first_created_epoch") or v["created"], "dur": dur,
                        "state": "running", "first": r.get("first_created_epoch") or v["created"], "since": v["created"]}
    for run, v in modal.items():
        if v["ended"] is False and not run.lower().startswith(skip):
            g = str(v.get("gpu") or ""); n, _, k = g.partition(":")
            run_now[run.lower()] = {"run": run, "platform": "modal", "gpu": f"{k}x {n}" if k else g or "?", "start": v["start"], "dur": v["timeout"],
                                    "state": v.get("state", "running"), "first": v.get("first"), "since": v.get("since")}
    for low, x in run_now.items():
        where, e = rows.get(low, (None, None))
        planned = None
        if e and e.get("start") and e.get("end") and not e.get("start_epoch"):
            try:
                a, b = [int(t.split()[0][:2]) * 60 + int(t.split()[0][3:5]) for t in (e["start"], e["end"])]; planned = ((b - a) % 1440 or 1440) * 60
            except Exception: planned = None  # an agent-written row with odd times: use the job's own timeout
        dur = x["dur"] or planned or 3600
        if x["platform"] == "modal" and planned: dur = min(dur, planned) if x["dur"] else planned  # the arm's planned length, within the timeout
        m = re.search(r"-(\d+)([mh])-", x["run"])  # play length in the name (-60m-, -4h-) plus 20 min setup caps a default 24 h timeout
        if m: dur = min(dur, int(m.group(1)) * (60 if m.group(2) == "m" else 3600) + 1200)
        if e and e.get("start_epoch") and e.get("end_epoch"): dur = e["end_epoch"] - e["start_epoch"]
        f = {"start": hm(x["start"]), "end": hm(x["start"] + dur), "start_epoch": round(x["start"]), "end_epoch": round(x["start"] + dur)}
        if where == "live":
            if any(e.get(k) != v for k, v in f.items()): ops.append(("set", e["run"], f))
        elif where == "queued": ops.append(("move", e["run"], f))
        else:
            arm = arm_of(x["run"], rules)
            ops.append(("add", x["run"], {"run": x["run"], "arm": arm, "platform": x["platform"], "gpu": x["gpu"], **f, "what": what_of(x["run"], arm, q, labels), "auto": True}))
    starting = {}
    for where in ("live", "queued"):
        for e in q.get(where, []):
            low = e["run"].lower()
            if low in run_now: continue
            if e.get("platform") == "nebius" and low in rec:
                r = json.loads(rec[low].read_text())
                t0 = r.get("first_created_epoch") or r.get("created_epoch") or os.path.getmtime(rec[low])
                if now - t0 < GRACE:
                    starting[e["run"]] = {"state": "starting", "first": None, "since": t0}
                    if where == "queued":  # launched: its receipt is there, the machine is not listed yet
                        dur = (r.get("spec") or {}).get("timeout") or r.get("play_seconds") or 3600
                        ops.append(("move", e["run"], {"start": hm(t0), "end": hm(t0 + dur), "start_epoch": round(t0), "end_epoch": round(t0 + dur)}))
                    continue
                if vms is not None: ops.append(("drop", e["run"], {"nebius": True}))
                elif where == "live" and ended_in_bucket(r.get("run_id") or e["run"]): ops.append(("drop", e["run"], {}))  # listing failed: the end marker
            elif e.get("platform") == "nebius" and where == "live" and vms is not None and e.get("auto"):
                ops.append(("drop", e["run"], {}))  # loop-added row (e.g. a collaborator VM) with no receipt: its machine is gone
            elif e.get("platform") == "kaggle" and where == "live" and now > (e.get("end_epoch") or now) + 2 * 3600:
                ops.append(("drop", e["run"], {}))  # Kaggle save run: no machine listing; gone 2 h past its planned end
            elif e.get("platform") != "nebius" and (modal.get(e["run"], {}).get("ended") or str(e.get("state", "")).lower() in END
                                                     or (where == "live" and app_ended(e.get("app_id")))):
                ops.append(("drop", e["run"], {}))  # Modal job or its own app ended, or the arm marked its row finished
    info = {**starting, **{x["run"]: {k: x.get(k) for k in ("state", "first", "since")} for x in run_now.values()}}
    return ops, info

def run(st, log=print):
    """One pass: returns the ops applied. Settles the ledger row of each dropped Nebius run (the launcher's status call)."""
    global MODAL_OK; MODAL_OK = True
    now = time.time(); vms = nebius_vms(); q = json.loads(Q.read_text())
    try: modal = modal_jobs(now)
    except Exception as exc: log("modal check failed:", exc); modal = {}; MODAL_OK = False
    if vms is None:
        log("Nebius listing failed: no Nebius rows added or dropped this pass")
    ops, info = plan(q, vms, modal, now, st)
    (paths.SCRATCH / "mon").mkdir(parents=True, exist_ok=True)
    (paths.SCRATCH / "mon/state.json").write_text(json.dumps({"t": now, "nebius_ok": vms is not None, "modal_ok": MODAL_OK, "runs": info}))
    if not ops: return info
    try: cards = json.loads((paths.SCRATCH / "mon/cards.json").read_text())
    except Exception: cards = {}
    with open(paths.SCRATCH / "mon/ended.jsonl", "a") as g:  # for the page's "ended in the last hour"
        for op, run_id, f in ops:
            if op == "drop":
                e = next((x for x in q.get("live", []) if x["run"] == run_id), None)
                if e: g.write(json.dumps({"run": run_id, "arm": e.get("arm"), "t": now, "final": cards.get(run_id, "")}) + "\n")
    sys.path.insert(0, str(ROOT / "src"))
    from arc_infra.runops import records  # the run records are the source; the queue file is their view (infra, 24 Oct)
    records.apply_queue_ops(ops)
    env = {**os.environ, "ARC_NEBIUS_CONFIG": str(paths.NEBIUS_CONFIG), "ARC_VALIDATION_ROOT": str(ROOT), "ARC_VALIDATION_KIND": "arena"}
    for op, run_id, f in ops:
        log("queue:", op, run_id, {k: v for k, v in f.items() if k in ("start", "end", "arm")})
        if op == "drop" and f.get("nebius"):
            try: subprocess.run([PY, str(paths.NEBIUS_SCRIPTS / "nebius_validation.py"), "status", "--run-id", run_id], capture_output=True, text=True, timeout=60, env=env)
            except subprocess.TimeoutExpired: pass  # settles at a later pass
    return info
