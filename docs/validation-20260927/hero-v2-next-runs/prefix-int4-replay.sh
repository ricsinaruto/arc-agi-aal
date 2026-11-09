#!/bin/sh
# CPU replay of the pinned scheduler at the int8 (1,600-token) and int4 (3,136-token) block geometries, scaled to
# 64-token sim blocks: int8 batch 4096/1600*64 = 164, system 4.1K -> 166; int4 batch 4096/3136*64 = 84, system -> 84,
# new/reply tokens scaled by 1600/3136.
S=<scratch>
SIM=./.claude/worktrees/agent-af122162397a4122e/scripts/arena_analysis/prefix_cache_sim.py
for ov in ctrl all; do
  for geo in int8 int4; do
    for regime in phase2 opening; do
      if [ $geo = int8 ]; then B="--batched 164 --system 166"; NEW="20 140"; REP="30 160"; DM=328; else B="--batched 84 --system 84"; NEW="10 71"; REP="15 82"; DM=167; fi
      if [ $regime = phase2 ]; then R="--games 6 --requests 12 --blocks 860 --gap 2"; else R="--games 24 --requests 12 --blocks 700 --gap 60"; fi
      extra=""; [ $ov = all ] && extra="--demote-min $DM"
      out=$(PYTHONPATH=$S/kv4pc/ov_$ov:$S/simdeps $S/torchvenv/bin/python $SIM run --block 64 $B $R --new-abs $NEW --reply-abs $REP $extra 2>&1 | tail -1)
      echo "$ov $geo $regime $out"
    done
  done
done
