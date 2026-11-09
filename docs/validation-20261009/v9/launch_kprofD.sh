#!/bin/bash
# INT8 vs BF16 dense and top-k 6 bench on one VM (+ ~13:50 UTC via coordinator). Usage: launch_kprofD.sh [VERSION]
cd .
RUN=b2-kprofD-dense-${1:-v1}
B=./.claude/worktrees/v84sub/dist/$RUN
export ARC_ARM=arc3sub
export REASON="decode speed of INT8 vs BF16 dense and of top-k 6 at 50 x 100K on p75 c613, same VM"
export APPROVAL="inside arc3sub's 2 RTX, under 2 h, expected -5"
export V87B_STACK=0 V87B_WHY="serving bench, no play: v9b stack, 50 forced decodes at ~100K, dense format or top-k the only difference"
WHAT="Dense format bench on one VM: p75 c613 at 50 real ~100K contexts with forced decodes, INT8 dense, then BF16 dense, then INT8 at top-k 6 (unhealed, throughput only), then INT8 again, the server restarted in place between arms. Why: we wanted the decode speed of INT8 dense against BF16, and top-k 6 against top-k 10, with the VM cancelled out. Judged by ms/step, steps/s, tok/s, blocks and KV GiB per arm; the repeat's gap is the noise."
bash <scratch>/limits/launch_rtx_appr.sh $RUN $B "$WHAT" 2>&1 | tail -6 | cut -c1-300
date -u; date +%s
