#!/bin/zsh
# dash_loop.sh: every 10 minutes reconcile the run queue with the machines, then regenerate MODAL_STATUS.md / .html and the arm
# status pages (docs/html) from the queue, the live feeds, the arm logs and the ledger (tools/ops/mon.py).
# Settings: runs/ops/ops.env (link to <scratch>/ops.env) with ARC_OPS_SCRATCH and ARC_NEBIUS_CONFIG; extra runs to print, one per line, in <scratch>/runs.txt.
ROOT=$(cd "$(dirname "$0")/../.." && pwd)
[ -f $ROOT/runs/ops/ops.env ] && source $ROOT/runs/ops/ops.env
export ARC_OPS_SCRATCH=${ARC_OPS_SCRATCH:-$ROOT/runs/ops}; S=$ARC_OPS_SCRATCH
[ -f $S/ops.env ] && source $S/ops.env
export ARC_NEBIUS_CONFIG=${ARC_NEBIUS_CONFIG:-$S/nebius.json}
mkdir -p "$S/launch" "$S/mon"; touch $S/runs.txt
PY=$ROOT/.venv/bin/python; [ -x $PY ] || PY=python3
while true; do
  t0=$(date +%s)
  # a crashed pass is recorded in mon/failed.txt; the next pass shows it in the dashboard check banner, a clean pass clears it
  if $PY $ROOT/tools/ops/mon.py $(cat $S/runs.txt) > $S/mon/last.tmp 2>&1; then rm -f $S/mon/failed.txt
  else { date -u +"%F %T UTC"; tail -3 $S/mon/last.tmp; } > $S/mon/failed.txt; fi
  mv $S/mon/last.tmp $S/mon/last.txt
  # the launcher and reconcile.py edit the ledger and queue in place: commit them each pass so agents can pull
  if ! git -C $ROOT diff --quiet -- docs/spend.json docs/modal-queue.json; then
    git -C $ROOT commit -q -m "Ledger/queue sync (dashboard pass)" -- docs/spend.json docs/modal-queue.json && git -C $ROOT pull --no-rebase -q --no-edit && git -C $ROOT push -q
  fi
  d=$(( 600 - $(date +%s) + t0 )); sleep $(( d > 30 ? d : 30 ))  # a pass every 10 minutes from its start, however long it took
done
