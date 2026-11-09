#!/bin/zsh
# dash_loop.sh: every 5 minutes reconcile the run queue with the machines, then regenerate MODAL_STATUS.md / .html and the arm
# status pages (docs/html) from the queue, the live feeds, the arm logs and the ledger (tools/ops/mon.py).
# Settings: runs/ops/ops.env (link to <scratch>/ops.env) with ARC_OPS_SCRATCH and ARC_NEBIUS_CONFIG; extra runs to print, one per line, in <scratch>/runs.txt.
ROOT=$(cd "$(dirname "$0")/../.." && pwd)
[ -f $ROOT/runs/ops/ops.env ] && source $ROOT/runs/ops/ops.env
export ARC_OPS_SCRATCH=${ARC_OPS_SCRATCH:-$ROOT/runs/ops}; S=$ARC_OPS_SCRATCH
[ -f $S/ops.env ] && source $S/ops.env
export ARC_NEBIUS_CONFIG=${ARC_NEBIUS_CONFIG:-$S/nebius.json}
mkdir -p "$S/launch" "$S/mon"; touch $S/runs.txt
PY=$ROOT/.venv/bin/python; [ -x $PY ] || PY=python3
# run records (docs/runs/<run>.json) and the views spend.json / modal-queue.json are machine-local and untracked (phase D,
# 24 Oct): sync regenerates the views (direct view edits are dropped and flagged); once a UTC day `snapshot` commits the
# ended records to docs/runs-settled/ and pushes
records_sync() { $PY $ROOT/tools/ops/records.py sync > $S/mon/records.txt 2>&1 || { date -u +"%F %T UTC records sync failed"; tail -3 $S/mon/records.txt; } >> $S/mon/records-failed.txt; }
while true; do
  t0=$(date +%s)
  records_sync
  H=$(( $(date +%s) / 3600 )); [ "$H" != "${BILLED_H:-}" ] && $PY $ROOT/tools/ops/records.py billing >> $S/mon/records.txt 2>&1 && BILLED_H=$H  # Modal's bill, hourly, for the $ gate
  [ "$H" != "${BAK_H:-}" ] && $PY $ROOT/tools/ops/records_backup.py backup >> $S/mon/records-backup.txt 2>&1 && BAK_H=$H  # records -> Nebius bucket, hourly, keep 48
  [ "$H" != "${VOL_H:-}" ] && { (cd $ROOT && $PY tools/ops/volume_usage.py > $S/mon/volume.txt 2>&1 &); VOL_H=$H; }  # runs volume inodes/bytes, hourly, background
  $PY $ROOT/tools/ops/ci.py poll >> $S/mon/ci.txt 2>&1  # CI: a suite run (Modal CPU, background) per new head of a watched branch
  # a crashed pass is recorded in mon/failed.txt; the next pass shows it in the dashboard check banner, a clean pass clears it
  if $PY $ROOT/tools/ops/mon.py $(cat $S/runs.txt) > $S/mon/last.tmp 2>&1; then rm -f $S/mon/failed.txt
  else { date -u +"%F %T UTC"; tail -3 $S/mon/last.tmp; } > $S/mon/failed.txt; fi
  mv $S/mon/last.tmp $S/mon/last.txt
  D=$(date -u +%F); G=$(git -C $ROOT rev-parse --absolute-git-dir)
  if [ "$D" != "${SNAP_D:-}" ] && [ ! -d $G/rebase-merge ] && [ ! -d $G/rebase-apply ] && [ ! -f $G/MERGE_HEAD ]; then
    $PY $ROOT/tools/ops/records.py snapshot >> $S/mon/records.txt 2>&1 && SNAP_D=$D  # settled records to git, daily
  fi
  d=$(( 300 - $(date +%s) + t0 )); sleep $(( d > 30 ? d: 30 ))  # a pass every 5 minutes from its start, however long it took
done
