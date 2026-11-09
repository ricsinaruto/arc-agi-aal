"""rec_attempt.py RUN: after a Nebius launch, give the run record what the launcher's reserve leaves out (27 Oct:
b2-qadv2-* had no attempts row, the dashboard flagged them): attempt 1 (running, start = the reserve's order), the
ledger row's start_epoch / app_id / bound, and a running queue row with the VM id; all through the records library."""
import json
import sys

sys.path.insert(0, "./src")
from arc_infra.runops import records  # noqa: E402

run = sys.argv[1]
nb = json.load(open(f"./docs/validation-nebius/nebius-{run}.json"))
iid = nb.get("instance_id")
if not iid:
    sys.exit(f"{run}: no instance_id yet")


def fn(rec):
    start = round(rec["ledger"][-1]["order"] / 1000)
    rec.setdefault("attempt", 1)
    if not rec.get("attempts"):
        rec["attempts"] = [{"n": rec["attempt"], "state": "running", "start_epoch": start}]
    row = rec["ledger"][-1]["row"]
    row.setdefault("start_epoch", start)
    row.setdefault("attempt", rec["attempt"])
    row["app_id"] = iid
    row["charged_bound_usd"] = round(float(row.get("charged_bound_usd") or nb.get("charged_bound_usd")), 4)


records.update(run, fn)
records.set_queue(run, app_id=iid, state="running")
rec = records.update(run, lambda r: None)
print(json.dumps({"run": run, "state": rec.get("state"), "attempts": rec.get("attempts"), "app_id": iid}))
records.publish()
