#!/bin/bash
# Layers-only decode bench on one VM. Usage: ARM=arc3sub|sft launch_kprofE.sh off|on [VERSION]
cd .
RUN=b2-kprofE-$1-${2:-v1}
B=./.claude/worktrees/v84sub/dist/$RUN
export ARC_ARM=${ARM:-arc3sub}
export REASON="clean layers-only decode bench, greedy 8 and 16 layers, graph list off and on, 50 x 100K on p75 c613, same VM; replaces the d4 x1.0 read"
export APPROVAL="+ ~14:55 UTC: arc3sub 2 RTX plus SFT idle 2 RTX; stop for review past 3 h"
export V87B_STACK=0 V87B_WHY="serving bench, no play: v9b INT8 stack, 50 forced decodes at ~100K, layers and graph list the only differences"
WHAT="Layers-only decode bench, graph list $1, on one VM: p75 c613 INT8 dense at 50 real ~100K contexts with forced decodes; base, 8 and 16 layers dropped, on list off also BF16 dense and a base repeat, the server restarted in place between arms. Unhealed, throughput only. Why: the maintainer replaces the d4 x1.0 read, which ran 64 requests on a 53-fit gate. Judged by ms/step, steps/s, tok/s, A, blocks, gate, KV GiB, GPU util and % FULL, PIECEWISE, eager per arm; the repeat gap is the noise."
bash <scratch>/limits/launch_rtx_appr.sh $RUN $B "$WHAT" 2>&1 | tail -6 | cut -c1-300
date -u; date +%s
