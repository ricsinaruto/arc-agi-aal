#!/bin/bash
# extreme p75 cut bench. Usage: launch_kprofC.sh anchor|cut [VERSION]
cd .
ARM=$1; RUN=b2-kprofC-$ARM-${2:-v1}
B=./.claude/worktrees/v84sub/dist/$RUN
export ARC_ARM=arc3sub
export REASON="throughput of the extreme p75 cut at 50 streams on 100K contexts, decode step and pool vs p75 c613 INT8 anchor; arm $ARM"
export APPROVAL="1 RTX inside the allowance of 2, under 2 h; two VMs approved"
export V87B_STACK=0 V87B_WHY="serving bench, no play: v9b INT8 stack, 50 forced decodes at ~100K, arm $ARM"
WHAT="Extreme p75 cut bench, arm $ARM: 50 streams held at ~100K real contexts with forced decodes on the v9b INT8 stack, no games. Why: we wanted the throughput of the 32-layer, 64-expert, top-k 6 cut against p75 c613. Judged by decode ms per step, steps/s, tok/s at A=2.45, pool blocks; prediction 22 ms and ~4,600 blocks for the cut."
bash <scratch>/limits/launch_rtx_appr.sh $RUN $B "$WHAT" 2>&1 | tail -6 | cut -c1-300
date -u; date +%s
