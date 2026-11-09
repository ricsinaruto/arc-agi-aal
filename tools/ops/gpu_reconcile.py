"""Reconcile live GPU use against docs/modal-queue.json: B300/RTX counts vs caps, unknown apps/VMs, stale rows.
usage: python3 tools/ops/gpu_reconcile.py [--b300-cap N] [--rtx-cap N] [--scope modal|nebius|all]
Exit 1 only when live GPUs are over a cap (a launch is never refused for another arm's bookkeeping).
Modal GPUs are counted from Modal itself (tools/ops/modal_gpus.py: each live container's GPU type), so an app without a
queue row still counts by its real GPU type; it is a NOTE. Rows are the fallback when the Modal API cannot be read."""
import argparse, json, os, re, subprocess, sys, time
import paths
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
NEBIUS = os.path.expanduser("~/.nebius/bin/nebius")

def gpus(s, kind):
    m = re.search(rf"(\d+)\s*x\s*{kind}", s or "", re.I)
    return int(m.group(1)) if m else (1 if kind.lower() in (s or "").lower() else 0)

def owner_hint(app):
    """'; likely <run> (records.py ledger <run> --set app_id=...)' for an arc-* app no record names, else ''."""
    try:
        import datetime
        sys.path.insert(0, os.path.join(ROOT, "src"))
        from arc_infra.runops import records
        t = datetime.datetime.fromisoformat(app["created_at"]).timestamp()
        run = records.likely_owner(app.get("description"), t)
    except Exception:  # noqa: BLE001 - a hint only
        return ""
    return f"; likely {run} (newest open record without an app id: records.py ledger {run} --set app_id='\"{app['app_id']}\"')" if run else ""


def modal_live():
    norm = paths.modal_apps("modal")
    if norm is None:
        return None  # unknown (listing failed or empty): callers must not read any app as gone
    return {a["app_id"]: a for a in norm if a["state"] != "stopped" and str(a["tasks"]) != "0"}

def nebius_live():
    if not os.path.exists(NEBIUS): return None
    r = subprocess.run([NEBIUS, "compute", "instance", "list", "--format", "json"], capture_output=True, text=True, timeout=60)
    try: items = json.loads(r.stdout).get("items", [])
    except Exception: return None
    return {i["metadata"]["name"]: i["status"].get("state") for i in items if i["status"].get("state") not in ("DELETING", "STOPPED")}

def modal_gpus():
    """{app_id: {"gpus": {...}}} from Modal (repo venv), or None when it cannot be read."""
    py = os.path.join(ROOT, ".venv/bin/python")
    if not os.path.exists(py):  # an arm worktree: the main checkout's venv
        common = subprocess.run(["git", "-C", ROOT, "rev-parse", "--git-common-dir"], capture_output=True, text=True).stdout.strip()
        py = os.path.join(os.path.dirname(os.path.abspath(os.path.join(ROOT, common))), ".venv/bin/python")
    try:
        r = subprocess.run([py if os.path.exists(py) else sys.executable, os.path.join(ROOT, "tools/ops/modal_gpus.py")],
                           capture_output=True, text=True, timeout=120)
        d = json.loads(r.stdout)
        return None if "error" in d else d
    except Exception:
        return None

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--b300-cap", type=int, default=8); ap.add_argument("--rtx-cap", type=int, default=10); ap.add_argument("--scope", choices=["modal", "nebius", "all"], default="all")
    a = ap.parse_args()
    qd = paths.read_json(f"{ROOT}/docs/modal-queue.json"); q = qd["live"]
    budget = paths.read_json(f"{ROOT}/docs/budget.json") or {}
    sys.path.insert(0, os.path.join(ROOT, "src"))
    from arc_infra.runops.records import is_external
    EXT = lambda run: is_external(run, budget)  # noqa: E731  collaborators' runs (budget.json external_runs): never counted
    q = [r for r in q if not EXT(r.get("run"))]
    cpu_apps = set(qd.get("cpu_apps", []))  # registered CPU-only apps
    try:  # the local checkout can lag origin by minutes: an arm's row pushed before its launch must count (23 Oct)
        subprocess.run(["git", "-C", ROOT, "fetch", "-q", "origin", "main"], timeout=30, capture_output=True)
        od = json.loads(subprocess.run(["git", "-C", ROOT, "show", "origin/main:docs/modal-queue.json"], capture_output=True, text=True, timeout=30).stdout)
        seen = {r.get("app_id") for r in q if r.get("app_id")}
        q = q + [r for r in od.get("live", []) if r.get("app_id") and r["app_id"] not in seen]
        cpu_apps |= set(od.get("cpu_apps", []))
    except Exception: pass
    cpu_names = set(qd.get("cpu_app_names", []))  # app names used only by CPU jobs: unregistered ones warn, not block (23 Oct)
    by_app = {r["app_id"]: r for r in q if r.get("app_id")}
    issues, b300, rtx = [], 0, 0  # (scope, text)
    apps = modal_live() if a.scope != "nebius" else {}  # a nebius launch must not hang on the Modal API (23 Oct)
    modal_known = apps is not None
    if not modal_known:
        issues.append(("note", "Modal API unreachable: app listing failed or empty; Modal GPUs counted from the queue rows"))
        apps = {}
    real = modal_gpus() if apps else {}
    for aid, app in apps.items():
        r = by_app.get(aid)
        live = (real or {}).get(aid, {}).get("gpus") if real is not None else None
        if live is not None:  # Modal's own count (registered or not)
            n = live.get("B300", 0) + live.get("B200", 0)
            if not r and aid not in cpu_apps:
                issues.append(("note", f"modal app {aid} ({app['description']}) has no queue row; counted as "
                                       f"{', '.join(f'{v} {k}' for k, v in live.items()) or 'CPU'}{owner_hint(app)}"))
        else:  # API unreadable: the row's GPUs; an unregistered app is a note, not a refusal
            if not r:
                if aid not in cpu_apps:
                    issues.append(("note", f"modal app {aid} ({app['description']}) has no queue row (GPU unknown){owner_hint(app)}"))
                continue
            n = gpus(r.get("gpu"), "B300") + gpus(r.get("gpu"), "B200")
        b300 += n
    if not modal_known:  # unknown listing: the live Modal rows' own GPUs
        b300 = sum(gpus(r.get("gpu"), "B300") + gpus(r.get("gpu"), "B200") for r in q
                   if r.get("platform") == "modal" and r.get("state", "running") == "running")
    for r in q:
        if modal_known and r.get("platform") == "modal" and r.get("app_id") and r.get("state", "running") == "running" and r["app_id"] not in apps:
            issues.append(("note", f"queue row {r['run']} says running but app {r['app_id']} is not live"))  # stale row uses no GPU: warn, don't block launches
    vms = nebius_live()
    if vms is not None:
        rows = {("arc-" + r["run"]).lower() for r in q if r.get("platform") == "nebius"}
        # long run names are truncated in the VM name (arc2 c240, 23 Oct): a long VM name matches a row it prefixes
        same = lambda vm, row: vm == row or (len(vm) >= 40 and row.startswith(vm))
        vml = [v.lower() for v in vms]
        for name in vms:
            if not name.startswith("arc-"): continue
            known = any(same(name.lower(), row) for row in rows)
            if re.match(r"arc-(b2-)?hx\d+-", name.lower()) or EXT(name[4:]):  # a collaborator's hx<N>- VM (harness is parked; the dash loop may add a row): outside our 8
                issues.append(("note", f"nebius VM {name} is a collaborator's (not counted)"))
                continue
            rtx += 1
            if not known: issues.append(("note", f"nebius VM {name} has no queue row (counted)"))
        for r in q:
            booting = time.time() - (r.get("start_epoch") or 0) < 600  # VM not listed yet
            if r.get("platform") == "nebius" and not any(same(v, ("arc-" + r["run"]).lower()) for v in vml) and r.get("state", "running") == "running" and not booting:
                issues.append(("note", f"queue row {r['run']} (nebius) has no live VM"))
    print(f"B300+B200 live {b300}/{a.b300_cap}, RTX VMs live {rtx}/{a.rtx_cap}")
    if b300 > a.b300_cap: issues.append(("modal", f"B300 over cap: {b300} > {a.b300_cap}"))
    if rtx > a.rtx_cap: issues.append(("nebius", f"RTX over cap: {rtx} > {a.rtx_cap}"))
    blocking = [t for sc, t in issues if sc != "note" and a.scope in ("all", sc)]
    for sc, t in issues: print("ISSUE" if t in blocking else "NOTE", t)
    sys.exit(1 if blocking else 0)

if __name__ == "__main__":
    main()
