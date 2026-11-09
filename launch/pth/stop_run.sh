#!/bin/bash
# stop_run.sh RUN "REASON" CAUSE: stop a Modal app or Nebius VM non-interactively and verify it is gone, then close its queue
# (and ledger) row via records.py settle --cause CAUSE (completed | stopped-owner | preempt | capacity | superseded |
# bug:<class>). Exit 0 only when verified stopped. Platform and id from the run's live queue row.
set -u
RUN=$1; WHY=${2:?reason}; CAUSE=${3:?cause: completed|stopped-owner|preempt|capacity|superseded|bug:<class>}; cd .; O=./.claude/worktrees/sft-heal/launch/pth; MV=.venv/bin/modal
row=$(python3 -c "import json,sys;r=[x for x in json.load(open('docs/modal-queue.json'))['live'] if x.get('run')=='$RUN'];print(json.dumps(r[-1]) if r else '')")
[ -n "$row" ] || { echo "stop_run: no queue row for $RUN"; exit 1; }
plat=$(echo "$row" | python3 -c "import json,sys;print(json.load(sys.stdin).get('platform',''))"); app=$(echo "$row" | python3 -c "import json,sys;print(json.load(sys.stdin).get('app_id',''))")
if [ "$plat" = nebius ]; then
  ARC_VALIDATION_ROOT=. ARC_VALIDATION_KIND=arena ARC_NEBIUS_CONFIG=./runs/ops/nebius.json .venv/bin/python - "$RUN" "$WHY" <<'PY' || exit 1
import json, sys, time
from pathlib import Path
sys.path.insert(0, "./.claude/worktrees/nebius-opd/scripts")
import nebius_validation as nv
run, why = sys.argv[1], sys.argv[2]
cloud = nv.Cloud(nv.load_config(), echo=lambda *a, **k: None)
rec = nv.load_record(Path("."), cloud, run)
current = (rec.get("attempts") or [{}])[-1].get("run_id", run)
cloud.put_text(json.dumps({"reason": why, "epoch": round(time.time(), 1)}) + "\n", nv.RUNS_PREFIX, "validation", current, "arena/live/stop-request")
iid = rec.get("instance_id")
if iid:
    cloud.delete(iid)
for _ in range(40):
    if iid not in [i["metadata"]["id"] for i in cloud.instances()]:
        print(json.dumps({"stopped": run, "instance": iid, "verified": True})); sys.exit(0)
    time.sleep(15)
print(json.dumps({"stopped": run, "instance": iid, "verified": False})); sys.exit(1)
PY
else
  [ -n "$app" ] || { echo "stop_run: no app id for $RUN"; exit 1; }
  $MV app stop -y "$app" >/dev/null 2>&1
  for i in $(seq 40); do st=$($MV app list --json 2>/dev/null | python3 -c "import json,sys;print([a['state'] for a in json.load(sys.stdin) if a['app_id']=='$app'])"); case "$st" in *stopped*|"[]") break;; esac; sleep 10; done
  case "$st" in *stopped*|"[]") echo "{\"stopped\": \"$RUN\", \"app\": \"$app\", \"verified\": true}";; *) echo "stop_run: $app still $st"; exit 1;; esac
fi
./.venv/bin/python ./tools/ops/records.py settle "$RUN" stopped --cause "$CAUSE" --set note="\"stopped: $WHY\"" && ./.venv/bin/python ./tools/ops/records.py push "$RUN"; exit 0
