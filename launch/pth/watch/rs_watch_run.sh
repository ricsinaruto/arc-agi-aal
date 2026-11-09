#!/bin/bash
# stops a replay VM still up 2 h after launch (hang guard for bundles built with the old 24 h timeout)
# 28 Oct: a VM that deleted itself after a failed bootstrap was reported as a plain end; final.json's state now decides
cd .
run=$1; deadline=$2
while [ "$(date +%s)" -lt "$deadline" ]; do
  if ./.claude/worktrees/sft-heal/launch/pth/vm_gone.sh $run; then
    fin=$(rclone cat nebius-arc:arc-runs-bucket/runs/validation/$run/nebius/final.json 2>/dev/null | tr -d '\n' | grep -o '"state": *"[a-z_]*"\|"error": *"[^"]*"' | tr '\n' ' ')
    echo "$fin" | grep -q '"state": *"\(complete\|preflight_ok\|baked\)"' && { echo "$run ended on its own: $fin"; exit 0; }
    echo "ALERT $run VM gone, final: ${fin:-unreadable}"; exit 1
  fi
  sleep 120
done
echo "ALERT $run still up at the deadline"; exit 1
