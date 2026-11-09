#!/bin/bash
# kprofR: per-step graph path log. Usage: launch_kprofR.sh h3|cut|anchor [VERSION]
cd .
ARM=$1; RUN=b2-kprofR-$ARM-${2:-v1}
B=./.claude/worktrees/v84sub/dist/$RUN
export ARC_ARM=arc3sub
export REASON="run_pw_graph investigation, per-step path, batch shape and CPU vs GPU time; arm $ARM"
export APPROVAL="owner yes via 1 RTX inside the allowance, under 2 h, measurement only"
export V87B_STACK=0 V87B_WHY="serving bench, no play: per-step path log, arm $ARM"
WHAT="kprofR, arm $ARM: logs every model step (graph path, batch shape, CPU and GPU time) on a play-like serving bench with no games. Why: about 19 ms of the 53 ms step sits in the piecewise graph path, and we asked what share of steps take it and what would remove it. Judged by the bars in the arc3sub arm log; no change to any live run."
bash <scratch>/limits/launch_rtx_appr.sh $RUN $B "$WHAT" 2>&1 | tail -6 | cut -c1-300
date -u; date +%s
