#!/bin/bash
# watch_rtx.sh RUN LAUNCH_EPOCH MINTOK: the RTX play watchers together; exits (waking the agent) on the first alert or end
run=$1; L=$2; M=${3:-1000}
cd ./.claude/worktrees/sft-heal/launch/pth/watch
if [ "$MODE" = replay ]; then ./replay_watch.sh $run $L >> watch-$run.log 2>&1 & a=$!; else ./play_watch.sh $run $L $M >> watch-$run.log 2>&1 & a=$!; fi
./rs_watch_run.sh $run $((L + ${HANG:-5400})) >> rs_watchdog-$run.log 2>&1 & b=$!
if [ "$MODE" = replay ]; then sleep 99999 & c=$!; else ./.claude/worktrees/sft-heal/launch/pth/play_metrics_watch.sh $run >> ~/arc-sft-data/mon/metrics-$run.log 2>&1 & c=$!; fi
while kill -0 $a 2>/dev/null && kill -0 $b 2>/dev/null && kill -0 $c 2>/dev/null; do sleep 30; done  # bash 3.2: no wait -n
kill $a $b $c 2>/dev/null
tail -2 watch-$run.log; tail -1 rs_watchdog-$run.log; tail -1 ~/arc-sft-data/mon/metrics-$run.log
