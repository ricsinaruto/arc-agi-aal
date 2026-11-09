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
# run records (docs/runs/<run>.json) are the source; spend.json / modal-queue.json are views. records.py sync imports what a
# legacy writer changed in the views since the last pass, then regenerates them (infra stage 4 cut-over, 24 Oct)
P="docs/spend.json docs/modal-queue.json docs/runs docs/budget.json"
records_sync() { $PY $ROOT/tools/ops/records.py sync > $S/mon/records.txt 2>&1 || { date -u +"%F %T UTC records sync failed"; tail -3 $S/mon/records.txt; } >> $S/mon/records-failed.txt; }
while true; do
  t0=$(date +%s)
  records_sync
  H=$(( $(date +%s) / 3600 )); [ "$H" != "${BILLED_H:-}" ] && $PY $ROOT/tools/ops/records.py billing >> $S/mon/records.txt 2>&1 && BILLED_H=$H  # Modal's bill, hourly, for the $ gate
  [ "$H" != "${VOL_H:-}" ] && { (cd $ROOT && $PY tools/ops/volume_usage.py > $S/mon/volume.txt 2>&1 &); VOL_H=$H; }  # runs volume inodes/bytes, hourly, background
  $PY $ROOT/tools/ops/ci.py poll >> $S/mon/ci.txt 2>&1  # CI: a suite run (Modal CPU, background) per new head of a watched branch
  # a crashed pass is recorded in mon/failed.txt; the next pass shows it in the dashboard check banner, a clean pass clears it
  if $PY $ROOT/tools/ops/mon.py $(cat $S/runs.txt) > $S/mon/last.tmp 2>&1; then rm -f $S/mon/failed.txt
  else { date -u +"%F %T UTC"; tail -3 $S/mon/last.tmp; } > $S/mon/failed.txt; fi
  mv $S/mon/last.tmp $S/mon/last.txt
  # the launcher and reconcile.py edit the ledger and queue in place: commit them each pass so agents can pull
  # never commit mid-rebase/merge or a file with conflict markers (21 Oct: a pass committed markers during an agent's rebase)
  G=$(git -C $ROOT rev-parse --absolute-git-dir)
  if [ ! -d $G/rebase-merge ] && [ ! -d $G/rebase-apply ] && [ ! -f $G/MERGE_HEAD ] \
     && $PY -c "import json,sys; [json.load(open(f)) for f in sys.argv[1:]]" $ROOT/docs/spend.json $ROOT/docs/modal-queue.json 2>/dev/null \
     && git -C $ROOT add -- docs/runs docs/budget.json && ! git -C $ROOT diff --quiet HEAD -- ${=P}; then
    git -C $ROOT commit -q -m "Ledger/queue sync (dashboard pass)" -- ${=P} && { git -C $ROOT pull --no-rebase -q --no-edit || { [ "$(git -C $ROOT diff --name-only --diff-filter=U)" = docs/modal-queue.json ] \
      && (cd $ROOT && $PY tools/ops/merge_queue.py >/dev/null && git add docs/modal-queue.json && git commit -q --no-edit); }; } \
      && { records_sync; git -C $ROOT add -- docs/runs docs/budget.json; git -C $ROOT diff --quiet HEAD -- ${=P} || git -C $ROOT commit -q -m "Records sync after pull" -- ${=P}; } \
      && git -C $ROOT push -q  # a queue-only conflict is merged now, not left open for agents to trip on (22 Oct)
    # a ledger conflict alongside the queue (23 Oct): take origin's ledger (reconcile.py re-applies this machine's settling next pass), merge the queue
    if [ -f $G/MERGE_HEAD ] && [ -z "$(git -C $ROOT diff --name-only --diff-filter=U | grep -v -e '^docs/spend.json$' -e '^docs/modal-queue.json$')" ]; then
      (cd $ROOT && { git diff --name-only --diff-filter=U | grep -q modal-queue && $PY tools/ops/merge_queue.py >/dev/null; true; } \
        && { git diff --name-only --diff-filter=U | grep -q spend.json && git checkout --theirs docs/spend.json; true; } \
        && $PY -c "import json,sys; [json.load(open(f)) for f in sys.argv[1:]]" docs/spend.json docs/modal-queue.json \
        && git add docs/spend.json docs/modal-queue.json && git commit -q --no-edit && git push -q)
    fi
    [ -f $G/MERGE_HEAD ] && git -C $ROOT merge --abort  # anything else: back out, the next pass pulls again
  fi
  d=$(( 300 - $(date +%s) + t0 )); sleep $(( d > 30 ? d: 30 ))  # a pass every 5 minutes from its start, however long it took
done
