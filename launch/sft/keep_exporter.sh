#!/bin/bash
# Keeps a run's dashboard exporter up. Exit 0 = the run ended; TERM = stopped by hand. Any other exit is a crash: it is
# restarted (its state file resumes it) and goes to watch.log, whose new lines wake the arm.
#   keep_exporter.sh PY EXPORTER RUN LOGDIR
PY=$1 EXP=$2 RUN=$3 LOG=$4
fast=0
while :; do
  began=$(date +%s)
  "$PY" -u "$EXP" "$RUN" --every 120 >> "$LOG/export.log" 2>&1
  rc=$?
  { [ "$rc" -eq 0 ] || [ "$rc" -eq 143 ]; } && exit 0
  if [ $(($(date +%s) - began)) -lt 600 ]; then fast=$((fast + 1)); else fast=0; fi
  echo "$(date -u +%H:%M) ALERT $RUN [exporter] exit $rc (crash $fast in a row): $(tail -1 "$LOG/export.log" | cut -c1-200)" >> "$LOG/watch.log"
  if [ "$fast" -ge 5 ]; then
    echo "$(date -u +%H:%M) ALERT $RUN [exporter] gave up after 5 quick crashes: dashboard is stale, fix and restart it" >> "$LOG/watch.log"
    exit 1
  fi
  sleep "${KEEP_EXPORTER_WAIT:-60}"
done
