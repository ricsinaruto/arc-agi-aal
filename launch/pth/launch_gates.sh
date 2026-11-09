#!/bin/bash
# launch_gates.sh: refusals before a B300/B200 heal round is submitted:
# live GPUs reconcile with the queue and caps, and every play a streaming round names is already writing rollouts.
cd .
out=$(python3 tools/ops/gpu_reconcile.py --b300-cap ${B300_CAP:-8} 2>&1); code=$?
[ $code = 0 ] || { echo "REFUSED: gpu_reconcile: $(echo "$out" | tail -4 | tr '\n' ' ')"; exit 1; }
# every student play writing, and at least one named base play (later base plays in a chain are pre-named, 22 Oct)
plays=$(echo "${ARC_HEAL_STREAM_PLAY:-}" | tr ',' ' ')
# ARC_HEAL_STREAM_PLAY_PENDING: named plays still booting while the round starts on the others' backlog
for p in $(echo "${ARC_HEAL_STREAM_PLAY_PENDING:-}" | tr ',' ' '); do plays=$(echo " $plays " | sed "s# $p # #g"); done
[ -z "${plays// }" ] || python3 ./.claude/worktrees/sft-heal/launch/pth/play_ready.py $plays > /dev/null || {
  echo "REFUSED: a student play has no rollout in the bucket yet ($plays)"; exit 1; }
bases=$(echo "${ARC_HEAL_STREAM_BASE:-}" | tr ',' ' ')
if [ -n "${bases// }" ]; then
  ok=0; for b in $bases; do python3 ./.claude/worktrees/sft-heal/launch/pth/play_ready.py $b > /dev/null 2>&1 && { ok=1; break; }; done
  [ $ok = 1 ] || { echo "REFUSED: no named base play has a rollout yet ($bases)"; exit 1; }
fi
exit 0
