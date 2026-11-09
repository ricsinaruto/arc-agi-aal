#!/bin/bash
# reload_check.sh RUN: the in-run checkpoint reload check of a streaming heal round (Modal CPU beside the round; owner
# 22 Oct: no GPU gate). Waits for the round's first checkpoint, compares it with the trainer's export of that step;
# prints RELOAD CHECK OK / FAILED (ALERT) when /runs/validation/RUN/reload-check.json lands.
RUN=$1; M=~/arc-sft-data/mon; MV=./.venv/bin/modal; T=$(mktemp -d)
$MV volume rm arc-prolong-runs /validation/$RUN/reload-check.json >/dev/null 2>&1  # a rerun never reads the last verdict
cd ./.claude/worktrees/sft-heal && $MV run --detach scripts/posttrain/heal_reload_check.py --run-id $RUN ${DELTA_RUN:+--delta-run $DELTA_RUN} ${DELTA_DIR:+--delta-dir $DELTA_DIR} > $M/reloadcheck-$RUN.log 2>&1
APP=$(grep -o 'ap-[A-Za-z0-9]\{20,\}' $M/reloadcheck-$RUN.log | head -1); echo "reload check app $APP"
[ -n "$APP" ] && (./.venv/bin/python ./tools/ops/records.py cpu-app $APP && ./.venv/bin/python ./tools/ops/records.py push $APP) >/dev/null 2>&1  # a CPU app row (records)
for i in $(seq 1100); do
  $MV volume get --force arc-prolong-runs /validation/$RUN/reload-check.json $T/r.json >/dev/null 2>&1 && break; sleep 20; done
python3 -I -c "import json,sys; d=json.load(open(sys.argv[1])); print(('RELOAD CHECK OK ' if d.get('ok') else 'ALERT RELOAD CHECK FAILED ') + json.dumps(d)[:1500])" $T/r.json 2>/dev/null || echo "ALERT reload check: no verdict"
