#!/bin/bash
# play_watch.sh RUN LAUNCH_EPOCH MIN_TOKS: every 5 min reads live_progress in the run's bootstrap.log; alerts and exits
# on a Traceback, no vLLM ready 50 min after launch, no live_progress for 15 min, or tok_s_recent below MIN_TOKS on 3
# reads in a row; logs running / KV / tok/s / completion tokens and the rollout files recorded.
# 23 Oct: "no vLLM ready" fired on two serving plays (bg1-v6, v9): a failed or partial read of bootstrap.log has no
# VLLM_SETUP_COMPLETE. Now: once ready is seen it stays ready; a failed/empty read changes nothing (3 in a row alert);
# the ready alert needs 2 readable reads past 50 min without it. Backtest: PW_LOGS=dir of snapshot files (read in
# order, one per 5-min tick from LAUNCH_EPOCH; an empty file = a failed read).
cd .
run=$1; launch=$2; min=$3; slow=0; last_e=""; last_t=$(date +%s); ready=0; noready=0; unread=0; tick=0
stopvm() { echo "$(date -u +%H:%M) ALERT $run: $1"; exit 1; }  # alert and exit, the agent decides
snaps=(); [ -n "${PW_LOGS:-}" ] && snaps=($(ls "$PW_LOGS" | sort -n)) && last_t=$launch
while true; do
  if [ -n "${PW_LOGS:-}" ]; then
    [ $tick -ge ${#snaps[@]} ] && { echo "BACKTEST END quiet"; exit 0; }
    log=$(cat "$PW_LOGS/${snaps[$tick]}"); rc=0; [ -z "$log" ] && rc=1; now=$((launch + 300 * (tick + 1))); tick=$((tick + 1))
  else
    if ./.claude/worktrees/sft-heal/launch/pth/vm_gone.sh $run; then echo "$(date -u +%H:%M) $run VM gone"; exit 0; fi
    log=$(rclone cat nebius-arc:arc-runs-bucket/runs/validation/$run/bootstrap.log 2>/dev/null); rc=$?; now=$(date +%s)
  fi
  if [ $rc -ne 0 ] || [ -z "$log" ]; then  # a failed read says nothing about the run
    unread=$((unread + 1)); [ $unread -ge 3 ] && stopvm "bootstrap.log unreadable on 3 reads"
    [ -n "${PW_LOGS:-}" ] || sleep 300; continue
  fi
  unread=0
  echo "$log" | grep -q "Traceback" && stopvm "Traceback"
  echo "$log" | grep -q "VLLM_SETUP_COMPLETE" && ready=1
  if [ $ready = 0 ]; then  # the 15-min stall clock starts at vLLM ready
    if [ $(( now - launch )) -gt 3000 ]; then noready=$((noready + 1)); [ $noready -ge 2 ] && stopvm "no vLLM ready at 50 min"; fi
    last_t=$now; [ -n "${PW_LOGS:-}" ] || sleep 300; continue
  fi
  lp=$(echo "$log" | grep '"live_progress"' | tail -1)
  if [ -n "$lp" ]; then
    e=$(echo "$lp" | python3 -c "import json,sys;r=json.loads(sys.stdin.read());print(r['elapsed_seconds'], r.get('tok_s_recent'), r.get('running'), round(r.get('kv') or 0,3), r.get('completion_tokens'), r.get('preemptions'))")
    set -- $e; [ "$1" != "$last_e" ] && { last_e=$1; last_t=$now; }
    rf=-; [ -n "${PW_LOGS:-}" ] || rf=$(rclone lsf -R --files-only --include "rollout.jsonl" nebius-arc:arc-runs-bucket/runs/validation/$run/arena/working/trials/ 2>/dev/null | wc -l)
    echo "$(date -u +%H:%M) elapsed $1 tok/s $2 running $3 kv $4 completion $5 preempt $6 rollout_files $rf"
    if python3 -c "import sys;sys.exit(0 if float('$2')<$min else 1)"; then slow=$((slow+1)); else slow=0; fi
    [ $slow -ge 3 ] && stopvm "tok/s below $min on 3 reads"
  fi
  [ $(( now - last_t )) -gt 900 ] && stopvm "no live_progress for 15 min"
  [ -n "${PW_LOGS:-}" ] || sleep 300
done
