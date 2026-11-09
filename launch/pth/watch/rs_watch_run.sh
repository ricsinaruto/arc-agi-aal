#!/bin/bash
# stops a replay VM still up 2 h after launch (hang guard for bundles built with the old 24 h timeout)
cd .
run=$1; deadline=$2
while [ "$(date +%s)" -lt "$deadline" ]; do
  ./.claude/worktrees/sft-heal/launch/pth/vm_gone.sh $run && { echo "$run ended on its own"; exit 0; }
  sleep 120
done
echo "ALERT $run still up at the deadline"; exit 1
