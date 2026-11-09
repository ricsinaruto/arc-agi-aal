#!/bin/bash
# Top-k 2 bench on one VM. Usage: launch_kprofF.sh [VERSION]
cd .
RUN=b2-kprofF-k2-${1:-v1}
B=./.claude/worktrees/v84sub/dist/$RUN
export ARC_ARM=arc3sub
export REASON="decode speed of top-k 2 at 50 x 100K on p75 c613, same VM as its base arms"
export APPROVAL="inside arc3sub 4 RTX"
export V87B_STACK=0 V87B_WHY="serving bench, no play: v9b INT8 stack, 50 forced decodes at ~100K, top-k the only difference"
WHAT="Top-k 2 bench on one VM: p75 c613 INT8 dense at 50 real ~100K contexts with forced decodes, base, then top-k 2 (unhealed, throughput only), then base again, the server restarted in place between arms. Why: we wanted the decode speed of top-k 2 with the VM cancelled out. Judged by ms/step, steps/s, tok/s, A, blocks, util, W and prefill against the base arms; the repeat gap is the noise."
bash <scratch>/limits/launch_rtx_appr.sh $RUN $B "$WHAT" 2>&1 | tail -6 | cut -c1-300
date -u; date +%s
